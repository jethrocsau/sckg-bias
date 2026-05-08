from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


class _ClassifierHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _ensure_known_mask(h_known_proj: Tensor, known_mask: Tensor | None) -> Tensor:
    if known_mask is None:
        return torch.ones(
            h_known_proj.shape[:2],
            device=h_known_proj.device,
            dtype=torch.bool,
        )
    return known_mask


def _masked_mean(x: Tensor, known_mask: Tensor) -> Tensor:
    known_mask_f = known_mask.unsqueeze(-1).to(x.dtype)
    return torch.sum(x * known_mask_f, dim=1) / known_mask_f.sum(dim=1).clamp_min(1.0)


def _pooled_kg_context(
    y_known: Tensor,
    known_mask: Tensor,
    h_kg_multi: Tensor | None,
    d_kg: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if h_kg_multi is None:
        return torch.zeros((y_known.shape[0], d_kg), device=device, dtype=dtype)
    kg_class = torch.mean(h_kg_multi, dim=1)
    kg_known = kg_class[y_known]
    return _masked_mean(kg_known, known_mask)


class MeanDecayRelationPool(nn.Module):
    def forward(self, h_kg_multi: Tensor) -> tuple[Tensor, Tensor]:
        num_rel = h_kg_multi.shape[1]
        pooled = torch.mean(h_kg_multi, dim=1)
        weights = torch.full(
            (h_kg_multi.shape[0], num_rel),
            1.0 / max(num_rel, 1),
            device=h_kg_multi.device,
            dtype=h_kg_multi.dtype,
        )
        return pooled, weights


class RelationGATPool(nn.Module):
    def __init__(self, d_kg: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_kg, hidden_dim, bias=False)
        self.attn_src = nn.Linear(hidden_dim, 1, bias=False)
        self.attn_dst = nn.Linear(hidden_dim, 1, bias=False)
        self.out_proj = nn.Linear(hidden_dim, d_kg, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h_kg_multi: Tensor) -> tuple[Tensor, Tensor]:
        # h_kg_multi: [C, R, D]
        h = self.proj(h_kg_multi)
        src_logits = self.attn_src(h)
        dst_logits = self.attn_dst(h)
        scores = self.leaky_relu(src_logits + dst_logits.transpose(1, 2))
        alpha = torch.softmax(scores, dim=-1)
        updated = torch.matmul(alpha, h)
        pooled_hidden = updated.mean(dim=1)
        relation_weights = alpha.mean(dim=1)
        return self.out_proj(pooled_hidden), relation_weights


class DGLRelationGATPool(nn.Module):
    def __init__(self, d_kg: int, hidden_dim: int):
        super().__init__()
        try:
            import dgl  # noqa: F401
            from dgl.nn import GATConv
        except Exception as exc:
            raise ImportError(
                "DGLRelationGATPool requires `dgl` to be installed. "
                "Install DGL or use --kg-embed-method mean_decay|gat instead."
            ) from exc

        self._dgl = __import__("dgl")
        self.gat = GATConv(
            in_feats=d_kg,
            out_feats=hidden_dim,
            num_heads=1,
            feat_drop=0.0,
            attn_drop=0.0,
            residual=False,
            allow_zero_in_degree=True,
        )
        self.out_proj = nn.Linear(hidden_dim, d_kg, bias=False)
        self._graph_cache: dict[tuple[int, str], object] = {}

    def _get_graph(self, num_rel: int, device: torch.device):
        key = (num_rel, str(device))
        graph = self._graph_cache.get(key)
        if graph is not None:
            return graph

        src = []
        dst = []
        for i in range(num_rel):
            for j in range(num_rel):
                src.append(i)
                dst.append(j)
        graph = self._dgl.graph((src, dst), num_nodes=num_rel, device=device)
        self._graph_cache[key] = graph
        return graph

    def forward(self, h_kg_multi: Tensor) -> tuple[Tensor, Tensor]:
        # h_kg_multi: [C, R, D]
        batch_size, num_rel, d_kg = h_kg_multi.shape
        base_graph = self._get_graph(num_rel, h_kg_multi.device)
        batched_graph = self._dgl.batch([base_graph] * batch_size)

        features = h_kg_multi.reshape(batch_size * num_rel, d_kg)
        updated, attn = self.gat(batched_graph, features, get_attention=True)
        updated = updated.squeeze(1).reshape(batch_size, num_rel, -1)

        pooled_hidden = updated.mean(dim=1)
        pooled = self.out_proj(pooled_hidden)

        attn = attn.squeeze(-1).squeeze(-1)
        relation_weights = torch.zeros(
            (batch_size, num_rel),
            device=h_kg_multi.device,
            dtype=h_kg_multi.dtype,
        )
        src_idx, dst_idx = batched_graph.edges()
        src_local = torch.remainder(src_idx, num_rel)
        batch_idx = torch.div(src_idx, num_rel, rounding_mode="floor")
        relation_weights.index_put_(
            (batch_idx, src_local),
            attn.to(h_kg_multi.dtype),
            accumulate=True,
        )
        relation_weights = relation_weights / relation_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return pooled, relation_weights


class _KGPoolMixin:
    def _init_kg_pool(self, cfg):
        self.kg_embed_method = getattr(cfg, "kg_embed_method", "mean_decay")
        if self.kg_embed_method == "gat":
            self.kg_pool = RelationGATPool(cfg.d_kg, cfg.d_model)
        elif self.kg_embed_method == "dgl_gat":
            self.kg_pool = DGLRelationGATPool(cfg.d_kg, cfg.d_model)
        else:
            self.kg_pool = MeanDecayRelationPool()

    def _pool_kg_per_class(self, h_kg_multi: Tensor | None, d_kg: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor | None]:
        if h_kg_multi is None:
            return torch.zeros((0, d_kg), device=device, dtype=dtype), None
        return self.kg_pool(h_kg_multi)

    def export_relation_weights(self, h_kg_multi: Tensor | None) -> Tensor | None:
        if h_kg_multi is None:
            return None
        _, weights = self.kg_pool(h_kg_multi)
        return weights.detach().cpu()


class TargetOnlyBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.target_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.target_ln = nn.LayerNorm(cfg.d_model)
        self.classifier = _ClassifierHead(
            input_dim=cfg.d_model,
            hidden_dim=cfg.d_model,
            num_classes=cfg.num_classes,
            dropout=getattr(cfg, "dropout", 0.3),
        )

    def forward(
        self,
        h0: Tensor,
        h_known: Tensor,
        y_known: Tensor,
        h_target: Tensor,
        known_mask: Tensor | None = None,
        h_kg_multi: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        del h0, h_known, y_known, known_mask, h_kg_multi
        h_target_proj = self.target_ln(self.target_proj(h_target))
        logits = self.classifier(h_target_proj)
        gate_weights = torch.ones((logits.shape[0], 1), device=logits.device, dtype=logits.dtype)
        return logits, {"gate_weights": gate_weights, "factor_weights": None}


class ContextOnlyBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.input_ln = nn.LayerNorm(cfg.d_model)
        self.classifier = _ClassifierHead(
            input_dim=cfg.d_model * 4,
            hidden_dim=cfg.d_model * 2,
            num_classes=cfg.num_classes,
            dropout=getattr(cfg, "dropout", 0.3),
        )

    def forward(
        self,
        h0: Tensor,
        h_known: Tensor,
        y_known: Tensor,
        h_target: Tensor,
        known_mask: Tensor | None = None,
        h_kg_multi: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        del y_known, h_target, h_kg_multi
        h0_proj = self.input_ln(self.input_proj(h0))
        h_known_proj = self.input_ln(self.input_proj(h_known))

        known_mask = _ensure_known_mask(h_known_proj, known_mask)
        context = _masked_mean(h_known_proj, known_mask)

        pair_repr = torch.cat(
            [
                h0_proj,
                context,
                h0_proj - context,
                h0_proj * context,
            ],
            dim=-1,
        )
        logits = self.classifier(pair_repr)
        gate_weights = torch.ones((logits.shape[0], 1), device=logits.device, dtype=logits.dtype)
        return logits, {"gate_weights": gate_weights, "factor_weights": None}


class KGContextBaseline(nn.Module, _KGPoolMixin):
    def __init__(self, cfg):
        super().__init__()
        self._init_kg_pool(cfg)
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.input_ln = nn.LayerNorm(cfg.d_model)
        self.target_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.target_ln = nn.LayerNorm(cfg.d_model)
        self.kg_proj = nn.Linear(cfg.d_kg, cfg.d_model)
        self.kg_ln = nn.LayerNorm(cfg.d_model)
        self.classifier = _ClassifierHead(
            input_dim=cfg.d_model * 5,
            hidden_dim=cfg.d_model * 2,
            num_classes=cfg.num_classes,
            dropout=getattr(cfg, "dropout", 0.3),
        )

    def forward(
        self,
        h0: Tensor,
        h_known: Tensor,
        y_known: Tensor,
        h_target: Tensor,
        known_mask: Tensor | None = None,
        h_kg_multi: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        h0_proj = self.input_ln(self.input_proj(h0))
        h_known_proj = self.input_ln(self.input_proj(h_known))
        h_target_proj = self.target_ln(self.target_proj(h_target))

        known_mask = _ensure_known_mask(h_known_proj, known_mask)
        context = _masked_mean(h_known_proj, known_mask)
        kg_class, relation_weights = self._pool_kg_per_class(
            h_kg_multi,
            self.kg_proj.in_features,
            h0.device,
            h0.dtype,
        )
        if kg_class.shape[0] == 0:
            kg_context = torch.zeros((h0.shape[0], self.kg_proj.in_features), device=h0.device, dtype=h0.dtype)
        else:
            kg_known = kg_class[y_known]
            kg_context = _masked_mean(kg_known, known_mask)

        kg_context_proj = self.kg_ln(self.kg_proj(kg_context))
        pair_repr = torch.cat(
            [
                h0_proj,
                h_target_proj,
                context,
                kg_context_proj,
                h_target_proj - context,
            ],
            dim=-1,
        )
        logits = self.classifier(pair_repr)
        gate_weights = torch.ones((logits.shape[0], 1), device=logits.device, dtype=logits.dtype)
        return logits, {"gate_weights": gate_weights, "factor_weights": None, "relation_weights": relation_weights}


class TargetPlusContextBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.input_ln = nn.LayerNorm(cfg.d_model)
        self.target_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.target_ln = nn.LayerNorm(cfg.d_model)
        self.classifier = _ClassifierHead(
            input_dim=cfg.d_model * 5,
            hidden_dim=cfg.d_model * 2,
            num_classes=cfg.num_classes,
            dropout=getattr(cfg, "dropout", 0.3),
        )

    def forward(self, h0: Tensor, h_known: Tensor, y_known: Tensor, h_target: Tensor, known_mask: Tensor | None = None, h_kg_multi: Tensor | None = None) -> tuple[Tensor, dict]:
        del y_known, h_kg_multi
        h0_proj = self.input_ln(self.input_proj(h0))
        h_known_proj = self.input_ln(self.input_proj(h_known))
        h_target_proj = self.target_ln(self.target_proj(h_target))
        known_mask = _ensure_known_mask(h_known_proj, known_mask)
        context = _masked_mean(h_known_proj, known_mask)
        pair_repr = torch.cat([h0_proj, h_target_proj, context, h_target_proj - context, h_target_proj * context], dim=-1)
        logits = self.classifier(pair_repr)
        gate_weights = torch.ones((logits.shape[0], 1), device=logits.device, dtype=logits.dtype)
        return logits, {"gate_weights": gate_weights, "factor_weights": None}


class TargetPlusKGBaseline(nn.Module, _KGPoolMixin):
    def __init__(self, cfg):
        super().__init__()
        self._init_kg_pool(cfg)
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.input_ln = nn.LayerNorm(cfg.d_model)
        self.target_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.target_ln = nn.LayerNorm(cfg.d_model)
        self.kg_proj = nn.Linear(cfg.d_kg, cfg.d_model)
        self.kg_ln = nn.LayerNorm(cfg.d_model)
        self.classifier = _ClassifierHead(
            input_dim=cfg.d_model * 5,
            hidden_dim=cfg.d_model * 2,
            num_classes=cfg.num_classes,
            dropout=getattr(cfg, "dropout", 0.3),
        )

    def forward(self, h0: Tensor, h_known: Tensor, y_known: Tensor, h_target: Tensor, known_mask: Tensor | None = None, h_kg_multi: Tensor | None = None) -> tuple[Tensor, dict]:
        del h_known
        h0_proj = self.input_ln(self.input_proj(h0))
        h_target_proj = self.target_ln(self.target_proj(h_target))
        if known_mask is None:
            known_mask = torch.ones(y_known.shape, device=y_known.device, dtype=torch.bool)
        kg_class, relation_weights = self._pool_kg_per_class(
            h_kg_multi,
            self.kg_proj.in_features,
            h0.device,
            h0.dtype,
        )
        if kg_class.shape[0] == 0:
            kg_context = torch.zeros((h0.shape[0], self.kg_proj.in_features), device=h0.device, dtype=h0.dtype)
        else:
            kg_known = kg_class[y_known]
            kg_context = _masked_mean(kg_known, known_mask)
        kg_context_proj = self.kg_ln(self.kg_proj(kg_context))
        pair_repr = torch.cat([h0_proj, h_target_proj, kg_context_proj, h_target_proj - kg_context_proj, h_target_proj * kg_context_proj], dim=-1)
        logits = self.classifier(pair_repr)
        gate_weights = torch.ones((logits.shape[0], 1), device=logits.device, dtype=logits.dtype)
        return logits, {"gate_weights": gate_weights, "factor_weights": None, "relation_weights": relation_weights}


class TargetPlusContextKGBaseline(nn.Module, _KGPoolMixin):
    def __init__(self, cfg):
        super().__init__()
        self._init_kg_pool(cfg)
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.input_ln = nn.LayerNorm(cfg.d_model)
        self.target_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.target_ln = nn.LayerNorm(cfg.d_model)
        self.kg_proj = nn.Linear(cfg.d_kg, cfg.d_model)
        self.kg_ln = nn.LayerNorm(cfg.d_model)
        self.classifier = _ClassifierHead(
            input_dim=cfg.d_model * 6,
            hidden_dim=cfg.d_model * 2,
            num_classes=cfg.num_classes,
            dropout=getattr(cfg, "dropout", 0.3),
        )

    def forward(self, h0: Tensor, h_known: Tensor, y_known: Tensor, h_target: Tensor, known_mask: Tensor | None = None, h_kg_multi: Tensor | None = None) -> tuple[Tensor, dict]:
        h0_proj = self.input_ln(self.input_proj(h0))
        h_known_proj = self.input_ln(self.input_proj(h_known))
        h_target_proj = self.target_ln(self.target_proj(h_target))
        known_mask = _ensure_known_mask(h_known_proj, known_mask)
        context = _masked_mean(h_known_proj, known_mask)
        kg_class, relation_weights = self._pool_kg_per_class(
            h_kg_multi,
            self.kg_proj.in_features,
            h0.device,
            h0.dtype,
        )
        if kg_class.shape[0] == 0:
            kg_context = torch.zeros((h0.shape[0], self.kg_proj.in_features), device=h0.device, dtype=h0.dtype)
        else:
            kg_known = kg_class[y_known]
            kg_context = _masked_mean(kg_known, known_mask)
        kg_context_proj = self.kg_ln(self.kg_proj(kg_context))
        pair_repr = torch.cat([
            h0_proj,
            h_target_proj,
            context,
            kg_context_proj,
            h_target_proj - context,
            h_target_proj - kg_context_proj,
        ], dim=-1)
        logits = self.classifier(pair_repr)
        gate_weights = torch.ones((logits.shape[0], 1), device=logits.device, dtype=logits.dtype)
        return logits, {"gate_weights": gate_weights, "factor_weights": None, "relation_weights": relation_weights}