from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


KGContextInput = Tensor | dict[str, Tensor] | None


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


def _extract_relation_context(h_kg_multi: KGContextInput) -> tuple[Tensor, Tensor | None]:
    if isinstance(h_kg_multi, dict):
        if "rel_embeddings" in h_kg_multi:
            return h_kg_multi["rel_embeddings"], h_kg_multi.get("rel_mask")
        if "path_embeddings" in h_kg_multi:
            return h_kg_multi["path_embeddings"], h_kg_multi.get("path_mask")
    return h_kg_multi, None


def _pack_ragged_relation_members(
    h_kg_multi: dict[str, Tensor],
    class_ids: Tensor | None = None,
) -> dict[str, Tensor] | None:
    if "member_offsets" not in h_kg_multi or "member_embeddings_flat" not in h_kg_multi:
        return None

    root_embeddings = h_kg_multi["root_embeddings"]
    member_offsets = h_kg_multi["member_offsets"]
    rel_mask = h_kg_multi.get("rel_mask")
    member_embeddings_flat = h_kg_multi["member_embeddings_flat"]
    member_weights_flat = h_kg_multi.get("member_weights_flat")

    if class_ids is not None:
        flat_class_ids = class_ids.reshape(-1)
        root_embeddings = root_embeddings[flat_class_ids]
        member_offsets = member_offsets[flat_class_ids]
        if rel_mask is not None:
            rel_mask = rel_mask[flat_class_ids]
    else:
        flat_class_ids = None

    if member_offsets.numel() == 0:
        num_rel = int(member_offsets.shape[-2]) if member_offsets.dim() >= 2 else 0
        flat_root = root_embeddings.reshape(-1, root_embeddings.shape[-1])
        return {
            "root_embeddings": flat_root.unsqueeze(1).expand(-1, num_rel, -1),
            "member_embeddings": root_embeddings.new_zeros((flat_root.shape[0], num_rel, 1, flat_root.shape[-1])),
            "member_mask": torch.zeros((flat_root.shape[0], num_rel, 1), device=root_embeddings.device, dtype=torch.bool),
            "member_weights": root_embeddings.new_zeros((flat_root.shape[0], num_rel, 1)),
            "rel_mask": rel_mask.reshape(-1, num_rel) if rel_mask is not None else torch.zeros((flat_root.shape[0], num_rel), device=root_embeddings.device, dtype=torch.bool),
        }

    num_rel = int(member_offsets.shape[-2])
    base_d_kg = int(root_embeddings.shape[-1])

    flat_offsets = member_offsets.reshape(-1, num_rel, 2)
    num_samples = flat_offsets.shape[0]
    counts = (flat_offsets[..., 1] - flat_offsets[..., 0]).reshape(-1).long()  # [N*R]
    max_members = int(counts.max().item()) if counts.numel() > 0 else 0
    max_members = max(max_members, 1)

    packed_members = root_embeddings.new_zeros((num_samples, num_rel, max_members, base_d_kg))
    packed_mask = torch.zeros((num_samples, num_rel, max_members), device=root_embeddings.device, dtype=torch.bool)
    packed_weights = root_embeddings.new_zeros((num_samples, num_rel, max_members))

    total = int(counts.sum().item())
    if total > 0:
        device = root_embeddings.device
        starts_flat = flat_offsets[..., 0].reshape(-1)  # [N*R]
        pair_ids = torch.repeat_interleave(
            torch.arange(num_samples * num_rel, device=device), counts
        )  
        cum_counts = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            counts.cumsum(0)[:-1],
        ])  # [N*R]
        pos_in_slot = (
            torch.arange(total, device=device) - torch.repeat_interleave(cum_counts, counts)
        )  

        global_flat_indices = starts_flat[pair_ids] + pos_in_slot  # [total]
        sample_ids = pair_ids // num_rel   # [total]
        rel_ids = pair_ids % num_rel       # [total]

        packed_members[sample_ids, rel_ids, pos_in_slot] = member_embeddings_flat[global_flat_indices]
        packed_mask[sample_ids, rel_ids, pos_in_slot] = True
        if member_weights_flat is not None:
            packed_weights[sample_ids, rel_ids, pos_in_slot] = member_weights_flat[global_flat_indices]
        else:
            packed_weights[sample_ids, rel_ids, pos_in_slot] = 1.0

    packed_root = root_embeddings.reshape(-1, 1, base_d_kg).expand(-1, num_rel, -1)
    packed_rel_mask = rel_mask.reshape(-1, num_rel) if rel_mask is not None else packed_mask.any(dim=-1)
    packed = {
        "root_embeddings": packed_root,
        "member_embeddings": packed_members,
        "member_mask": packed_mask,
        "member_weights": packed_weights,
        "rel_mask": packed_rel_mask,
    }

    if flat_class_ids is not None:
        packed["class_ids"] = flat_class_ids
    return packed


def _delta_mean(h0_proj: Tensor, h_known_proj: Tensor, known_mask: Tensor) -> Tensor:
    delta = h_known_proj - h0_proj.unsqueeze(1)
    return _masked_mean(delta, known_mask)


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
    def forward(self, h_kg_multi: KGContextInput) -> tuple[Tensor, Tensor]:
        rel_embeddings, rel_mask = _extract_relation_context(h_kg_multi)
        num_rel = rel_embeddings.shape[1]
        if rel_mask is None:
            rel_mask = torch.ones(
                rel_embeddings.shape[:2],
                device=rel_embeddings.device,
                dtype=torch.bool,
            )
        rel_mask_f = rel_mask.unsqueeze(-1).to(rel_embeddings.dtype)
        pooled = torch.sum(rel_embeddings * rel_mask_f, dim=1) / rel_mask_f.sum(dim=1).clamp_min(1.0)
        weights = rel_mask.to(rel_embeddings.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return pooled, weights


class MeanDecaySubtreePool(nn.Module):
    def __init__(self, input_dim: int, output_dim: int | None = None):
        super().__init__()
        if output_dim is None or output_dim == input_dim:
            self.out_proj = nn.Identity()
        else:
            self.out_proj = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, relation_members: dict[str, Tensor]) -> Tensor:
        root_embeddings = relation_members["root_embeddings"]
        member_embeddings = relation_members["member_embeddings"]
        member_mask = relation_members.get("member_mask")
        member_weights = relation_members.get("member_weights")

        if member_mask is None:
            member_mask = torch.ones(
                member_embeddings.shape[:-1],
                device=member_embeddings.device,
                dtype=torch.bool,
            )
        if member_weights is None:
            member_weights = member_mask.to(member_embeddings.dtype)

        masked_weights = member_weights * member_mask.to(member_weights.dtype)
        pooled_members = torch.sum(member_embeddings * masked_weights.unsqueeze(-1), dim=2)
        pooled_members = pooled_members / masked_weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
        if root_embeddings.dim() == 2:
            root_embeddings = root_embeddings.unsqueeze(1).expand(-1, pooled_members.shape[1], -1)
        return self.out_proj(root_embeddings + pooled_members)


class RelationSubtreeGATPool(nn.Module):
    def __init__(self, base_d_kg: int, hidden_dim: int, output_dim: int | None = None):
        super().__init__()
        self.root_proj = nn.Linear(base_d_kg, hidden_dim, bias=False)
        self.member_proj = nn.Linear(base_d_kg, hidden_dim, bias=False)
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        if output_dim is None or output_dim == base_d_kg:
            self.out_proj = nn.Identity()
        else:
            self.out_proj = nn.Linear(base_d_kg, output_dim, bias=False)

    def forward(self, relation_members: dict[str, Tensor]) -> Tensor:
        root_embeddings = relation_members["root_embeddings"]
        member_embeddings = relation_members["member_embeddings"]
        member_mask = relation_members.get("member_mask")
        member_weights = relation_members.get("member_weights")

        if root_embeddings.dim() == 2:
            root_embeddings = root_embeddings.unsqueeze(1).expand(-1, member_embeddings.shape[1], -1)

        root_hidden = self.root_proj(root_embeddings).unsqueeze(2)
        member_hidden = self.member_proj(member_embeddings)
        logits = self.score(torch.tanh(root_hidden + member_hidden)).squeeze(-1)

        if member_weights is not None:
            logits = logits + member_weights.clamp_min(1e-8).log()

        if member_mask is not None:
            logits = logits.masked_fill(~member_mask, float("-inf"))
            no_valid = ~member_mask.any(dim=2)
            if no_valid.any():
                logits = logits.clone()
                logits[no_valid] = 0.0

        weights = torch.softmax(logits, dim=-1)
        if member_mask is not None:
            weights = weights * member_mask.to(weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        pooled_members = torch.sum(member_embeddings * weights.unsqueeze(-1), dim=2)
        return self.out_proj(root_embeddings + pooled_members)


class HierarchicalRelationPool(nn.Module):
    def __init__(self, d_kg: int, hidden_dim: int, subtree_pool_method: str, relation_pool_method: str):
        super().__init__()
        if d_kg % 2 != 0:
            raise ValueError("Hierarchical relation pooling expects an even d_kg = 2 * base_d_kg.")

        base_d_kg = d_kg // 2
        if subtree_pool_method == "gat":
            self.subtree_pool = RelationSubtreeGATPool(base_d_kg, hidden_dim, output_dim=d_kg)
        else:
            self.subtree_pool = MeanDecaySubtreePool(base_d_kg, output_dim=d_kg)

        if relation_pool_method == "mean_decay":
            self.relation_pool = MeanDecayRelationPool()
        else:
            self.relation_pool = RelationGATPool(d_kg, hidden_dim)

    def forward(self, h_kg_multi: KGContextInput) -> tuple[Tensor, Tensor]:
        if isinstance(h_kg_multi, dict):
            packed_members = _pack_ragged_relation_members(h_kg_multi)
            if packed_members is not None:
                rel_embeddings = self.subtree_pool(packed_members)
                return self.relation_pool(
                    {
                        "rel_embeddings": rel_embeddings,
                        "rel_mask": packed_members.get("rel_mask"),
                    }
                )
            if "member_embeddings" in h_kg_multi and "root_embeddings" in h_kg_multi:
                rel_embeddings = self.subtree_pool(h_kg_multi)
                return self.relation_pool(
                    {
                        "rel_embeddings": rel_embeddings,
                        "rel_mask": h_kg_multi.get("rel_mask"),
                    }
                )
        return self.relation_pool(h_kg_multi)


class RelationGATPool(nn.Module):
    def __init__(self, d_kg: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_kg, hidden_dim, bias=False)
        self.attn_src = nn.Linear(hidden_dim, 1, bias=False)
        self.attn_dst = nn.Linear(hidden_dim, 1, bias=False)
        self.out_proj = nn.Linear(hidden_dim, d_kg, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h_kg_multi: KGContextInput) -> tuple[Tensor, Tensor]:
        rel_embeddings, rel_mask = _extract_relation_context(h_kg_multi)
        h = self.proj(rel_embeddings)
        src_logits = self.attn_src(h)
        dst_logits = self.attn_dst(h)
        scores = self.leaky_relu(src_logits + dst_logits.transpose(1, 2))
        if rel_mask is not None:
            dst_mask = rel_mask.unsqueeze(1)
            scores = scores.masked_fill(~dst_mask, float("-inf"))
            no_valid = ~rel_mask.any(dim=1)
            if no_valid.any():
                scores = scores.clone()
                scores[no_valid] = 0.0
        alpha = torch.softmax(scores, dim=-1)
        if rel_mask is not None:
            src_mask = rel_mask.unsqueeze(-1).to(alpha.dtype)
            alpha = alpha * src_mask
        updated = torch.matmul(alpha, h)
        if rel_mask is None:
            pooled_hidden = updated.mean(dim=1)
            relation_weights = alpha.mean(dim=1)
        else:
            src_mask_f = rel_mask.unsqueeze(-1).to(updated.dtype)
            pooled_hidden = torch.sum(updated * src_mask_f, dim=1) / src_mask_f.sum(dim=1).clamp_min(1.0)
            relation_weights = torch.sum(alpha * src_mask_f, dim=1) / src_mask_f.sum(dim=1).clamp_min(1.0)
            relation_weights = relation_weights * rel_mask.to(relation_weights.dtype)
            relation_weights = relation_weights / relation_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return self.out_proj(pooled_hidden), relation_weights


class PathAttentionPool(nn.Module):
    def __init__(self, d_kg: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_kg, hidden_dim, bias=False)
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        self.out_proj = nn.Linear(hidden_dim, d_kg, bias=False)

    def forward(self, h_kg_paths: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        path_embeddings = h_kg_paths["path_embeddings"]
        path_mask = h_kg_paths.get("path_mask")

        hidden = self.proj(path_embeddings)
        logits = self.score(hidden).squeeze(-1)
        if path_mask is not None:
            logits = logits.masked_fill(~path_mask, float("-inf"))
            no_valid = ~path_mask.any(dim=1)
            if no_valid.any():
                logits = logits.clone()
                logits[no_valid] = 0.0

        weights = torch.softmax(logits, dim=-1)
        if path_mask is not None:
            weights = weights * path_mask.to(weights.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        pooled_hidden = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        return self.out_proj(pooled_hidden), weights


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
        if self.kg_embed_method == "path_attn":
            self.kg_pool = PathAttentionPool(cfg.d_kg, cfg.d_model)
        else:
            subtree_pool_method = getattr(cfg, "kg_subtree_pool_method", "match")
            if subtree_pool_method == "match":
                subtree_pool_method = "gat" if self.kg_embed_method in {"gat", "dgl_gat"} else "mean_decay"
            relation_pool_method = getattr(cfg, "kg_relation_pool_method", "gat")
            self.kg_pool = HierarchicalRelationPool(
                cfg.d_kg,
                cfg.d_model,
                subtree_pool_method=subtree_pool_method,
                relation_pool_method=relation_pool_method,
            )

    def _pool_kg_per_class(self, h_kg_multi: KGContextInput, d_kg: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor | None]:
        if h_kg_multi is None:
            return torch.zeros((0, d_kg), device=device, dtype=dtype), None
        return self.kg_pool(h_kg_multi)

    def export_relation_weights(self, h_kg_multi: KGContextInput) -> Tensor | None:
        if h_kg_multi is None:
            return None
        _, weights = self.kg_pool(h_kg_multi)
        return weights.detach().cpu()


class _SharedBaselineMLP(nn.Module, _KGPoolMixin):
    def __init__(self, cfg, *, use_target: bool, use_context: bool, use_kg: bool):
        super().__init__()
        self.use_target = use_target
        self.use_context = use_context
        self.use_kg = use_kg

        self._init_kg_pool(cfg)
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.input_ln = nn.LayerNorm(cfg.d_model)
        self.target_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.target_ln = nn.LayerNorm(cfg.d_model)
        self.kg_proj = nn.Linear(cfg.d_kg, cfg.d_model)
        self.kg_ln = nn.LayerNorm(cfg.d_model)

        self.null_target = nn.Parameter(torch.zeros(cfg.d_model))
        self.null_context = nn.Parameter(torch.zeros(cfg.d_model))
        self.null_kg = nn.Parameter(torch.zeros(cfg.d_model))

        self.classifier = _ClassifierHead(
            input_dim=cfg.d_model * 4,
            hidden_dim=cfg.d_model * 2,
            num_classes=cfg.num_classes,
            dropout=getattr(cfg, "dropout", 0.3),
        )

    def _expand_null(self, param: nn.Parameter, batch_size: int) -> Tensor:
        return param.unsqueeze(0).expand(batch_size, -1)

    def export_relation_weights(self, h_kg_multi: KGContextInput) -> Tensor | None:
        if not self.use_kg:
            return None
        return super().export_relation_weights(h_kg_multi)

    def _pool_known_kg(
        self,
        y_known: Tensor,
        known_mask: Tensor,
        h_kg_multi: KGContextInput,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor | None]:
        if h_kg_multi is None:
            return torch.zeros((0, self.kg_proj.in_features), device=device, dtype=dtype), None

        if isinstance(h_kg_multi, dict) and "member_offsets" in h_kg_multi:
            packed_known = _pack_ragged_relation_members(h_kg_multi, y_known)
            if packed_known is None:
                return torch.zeros((0, self.kg_proj.in_features), device=device, dtype=dtype), None
            batch_size, num_known = y_known.shape
            flat_input = {
                key: value
                for key, value in packed_known.items()
                if key != "class_ids"
            }
            pooled_flat, relation_weights = self.kg_pool(flat_input)
            pooled_known = pooled_flat.reshape(batch_size, num_known, -1)
            pooled = _masked_mean(pooled_known, known_mask)
            if relation_weights is not None:
                relation_weights = relation_weights.reshape(batch_size, num_known, -1)
                mask_f = known_mask.unsqueeze(-1).to(relation_weights.dtype)
                relation_weights = torch.sum(relation_weights * mask_f, dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
            return pooled, relation_weights

        if isinstance(h_kg_multi, dict) and "rel_embeddings" in h_kg_multi:
            rel_embeddings = h_kg_multi["rel_embeddings"][y_known]
            rel_mask = h_kg_multi.get("rel_mask")
            gathered_mask = rel_mask[y_known] if rel_mask is not None else None
            batch_size, num_known, num_rel, d_kg = rel_embeddings.shape
            flat_input = {"rel_embeddings": rel_embeddings.reshape(batch_size * num_known, num_rel, d_kg)}
            if gathered_mask is not None:
                flat_input["rel_mask"] = gathered_mask.reshape(batch_size * num_known, num_rel)
            pooled_flat, relation_weights = self.kg_pool(flat_input)
            pooled_known = pooled_flat.reshape(batch_size, num_known, d_kg)
            pooled = _masked_mean(pooled_known, known_mask)
            if relation_weights is not None:
                relation_weights = relation_weights.reshape(batch_size, num_known, num_rel)
                mask_f = known_mask.unsqueeze(-1).to(relation_weights.dtype)
                relation_weights = torch.sum(relation_weights * mask_f, dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
            return pooled, relation_weights

        kg_class, relation_weights = self._pool_kg_per_class(
            h_kg_multi,
            self.kg_proj.in_features,
            device,
            dtype,
        )
        if kg_class.shape[0] == 0:
            return torch.zeros((y_known.shape[0], self.kg_proj.in_features), device=device, dtype=dtype), relation_weights
        kg_known = kg_class[y_known]
        return _masked_mean(kg_known, known_mask), relation_weights

    def forward(
        self,
        h0: Tensor,
        h_known: Tensor,
        y_known: Tensor,
        h_target: Tensor,
        known_mask: Tensor | None = None,
        h_kg_multi: KGContextInput = None,
    ) -> tuple[Tensor, dict]:
        batch_size = h_target.shape[0]

        target_repr = self._expand_null(self.null_target, batch_size)
        if self.use_target:
            target_repr = self.target_ln(self.target_proj(h_target))

        h0_proj = self.input_ln(self.input_proj(h0))

        mask_for_pool = known_mask
        context_repr = self._expand_null(self.null_context, batch_size)
        if self.use_context:
            h_known_proj = self.input_ln(self.input_proj(h_known))
            mask_for_pool = _ensure_known_mask(h_known_proj, known_mask)
            context_repr = _delta_mean(h0_proj, h_known_proj, mask_for_pool)

        if mask_for_pool is None:
            mask_for_pool = torch.ones(y_known.shape, device=y_known.device, dtype=torch.bool)

        relation_weights = None
        kg_repr = self._expand_null(self.null_kg, batch_size)
        if self.use_kg:
            kg_context, relation_weights = self._pool_known_kg(
                y_known,
                mask_for_pool,
                h_kg_multi,
                h_target.device,
                h_target.dtype,
            )
            if kg_context.shape[0] > 0:
                kg_repr = self.kg_ln(self.kg_proj(kg_context))

        pair_repr = torch.cat(
            [
                h0_proj,
                target_repr,
                context_repr,
                kg_repr,
            ],
            dim=-1,
        )
        logits = self.classifier(pair_repr)
        gate_weights = torch.ones((logits.shape[0], 1), device=logits.device, dtype=logits.dtype) # this is redundant in our task, leaving this here due to old design and compataibiltiy
        return logits, {"gate_weights": gate_weights, "factor_weights": None, "relation_weights": relation_weights}


class TargetOnlyBaseline(_SharedBaselineMLP):
    def __init__(self, cfg):
        super().__init__(cfg, use_target=True, use_context=False, use_kg=False)


class ContextOnlyBaseline(_SharedBaselineMLP):
    def __init__(self, cfg):
        super().__init__(cfg, use_target=False, use_context=True, use_kg=False)


class KGContextBaseline(_SharedBaselineMLP):
    def __init__(self, cfg):
        super().__init__(cfg, use_target=False, use_context=False, use_kg=True)


class TargetPlusContextBaseline(_SharedBaselineMLP):
    def __init__(self, cfg):
        super().__init__(cfg, use_target=True, use_context=True, use_kg=False)


class TargetPlusKGBaseline(_SharedBaselineMLP):
    def __init__(self, cfg):
        super().__init__(cfg, use_target=True, use_context=False, use_kg=True)


class ContextPlusKGBaseline(_SharedBaselineMLP):
    def __init__(self, cfg):
        super().__init__(cfg, use_target=False, use_context=True, use_kg=True)


class TargetPlusContextKGBaseline(_SharedBaselineMLP):
    def __init__(self, cfg):
        super().__init__(cfg, use_target=True, use_context=True, use_kg=True)