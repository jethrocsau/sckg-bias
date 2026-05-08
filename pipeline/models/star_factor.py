import torch
from torch import Tensor, nn
from .local_star_gnn import CellSinkGNN

class StarFactorModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.kg_enabled = getattr(cfg, 'kg_enabled', True)
        self.num_factors = getattr(cfg, 'num_factors', 4)
        p_dropout = getattr(cfg, 'dropout', 0.1)
        
        # Initial Projection + LayerNorm (Critical for multimodal stability)
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.input_ln = nn.LayerNorm(cfg.d_model)
        self.target_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.target_ln = nn.LayerNorm(cfg.d_model)
        
        self.gnn = CellSinkGNN(
            d_model=cfg.d_model, 
            d_msg=cfg.d_msg, 
            d_kg=cfg.d_kg, 
            num_layers=cfg.gat_depth,
            dropout=p_dropout
        )
        
        if self.kg_enabled:
            self.factor_weights = nn.Parameter(
                torch.randn(self.num_factors, cfg.num_classes, cfg.num_kg_relations)
            )
        else:
            self.null_kg_emb = nn.Parameter(torch.randn(self.num_factors, cfg.d_kg))
            
        # Deeper Router with LayerNorm
        self.moe_router = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, self.num_factors)
        )
        
        self.logit_head = nn.Sequential(
            nn.Linear(cfg.d_model * 4, cfg.d_model * 2),
            nn.LayerNorm(cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(cfg.d_model, cfg.num_classes)
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
        # 0. Initial Projection with Normalization
        h0_proj = self.input_ln(self.input_proj(h0))
        h_known_proj = self.input_ln(self.input_proj(h_known))
        h_target_proj = self.target_ln(self.target_proj(h_target))

        if known_mask is not None:
            h_known_proj = h_known_proj * known_mask.unsqueeze(-1)
        
        # 1. Factor Selection
        if self.kg_enabled and h_kg_multi is not None:
            alpha_rel = torch.softmax(self.factor_weights, dim=-1)
            h_kg_class_factors = torch.einsum('fcr, crd -> fcd', alpha_rel, h_kg_multi)
            h_kg_known = h_kg_class_factors.transpose(0, 1)[y_known]
        else:
            alpha_rel = None
            h_kg_known = self.null_kg_emb.view(1, 1, self.num_factors, -1).expand(h0.shape[0], h_known.shape[1], -1, -1)

        if known_mask is not None:
            h_kg_known = h_kg_known * known_mask.unsqueeze(-1).unsqueeze(-1)

        # 2. Virtual Cell Sink processing
        h0_factors = self.gnn(h0_proj, h_known_proj, h_kg_known, known_mask=known_mask)
        
        # 3. MoE Gating
        router_input = h0_proj + h_target_proj
        gate_weights = torch.softmax(self.moe_router(router_input), dim=-1)
        h0_final = torch.sum(gate_weights.unsqueeze(-1) * h0_factors, dim=1)
        pair_repr = torch.cat(
            [
                h0_final,
                h_target_proj,
                h0_final - h_target_proj,
                h0_final * h_target_proj,
            ],
            dim=-1,
        )
        
        aux_data = {
            "gate_weights": gate_weights,
            "factor_weights": alpha_rel
        }
        
        return self.logit_head(pair_repr), aux_data