"""Virtual Cell Sink convolution with parallel Factor processing."""

from __future__ import annotations
import torch
from torch import Tensor, nn

class ResidualMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
            nn.Dropout(dropout)
        )
        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.shortcut(x) + self.net(x)

class _CellSinkLayer(nn.Module):
    def __init__(self, d_model: int, d_msg: int, d_kg: int, dropout: float = 0.1):
        super().__init__()
        d_concat = d_model + d_kg
        
        # Attention: Drug querying the cellular environment to weight neighbors
        self.attn_mlp = nn.Sequential(
            nn.Linear(d_model + d_concat, d_msg), 
            nn.LeakyReLU(0.2),
            nn.Linear(d_msg, 1)
        )
        
        # Stage 1: Processes the neighbor+KG features into a Cell State
        self.nb_to_cell_mlp = ResidualMLP(d_concat, d_msg, d_msg, dropout)
        
        # Stage 2: Fuses the aggregated Cell State with the Drug State
        self.cell_drug_fusion = nn.Sequential(
            nn.Linear(d_msg + d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.ln = nn.LayerNorm(d_model)

    def forward(self, h0: Tensor, h_known: Tensor, h_kg: Tensor, known_mask: Tensor | None = None) -> Tensor:
        bsz, n_nb, num_factors, d_kg = h_kg.shape
        
        # 1. Expand known neighbors and concat with multimodal KG embeddings
        # h_known: [B, N, d_model] -> [B, N, F, d_model]
        h_known_ext = h_known.unsqueeze(2).expand(-1, -1, num_factors, -1)
        h_feat = torch.cat([h_known_ext, h_kg], dim=-1) # [B, N, F, d_model + d_kg]
        
        # 2. Attention: Drug attends to the cellular context
        h0_ext = h0.unsqueeze(1).expand(-1, n_nb, -1, -1) # [B, N, F, d_model]
        attn_input = torch.cat([h0_ext, h_feat], dim=-1)
        attn_logits = self.attn_mlp(attn_input).squeeze(-1)
        if known_mask is not None:
            mask = known_mask.unsqueeze(-1).expand(-1, -1, num_factors)
            attn_logits = attn_logits.masked_fill(~mask, -1e9)
        alpha = torch.softmax(attn_logits, dim=1) # [B, N, F]
        
        # 3. Virtual Cell Sink: Aggregate the environment
        m = self.nb_to_cell_mlp(h_feat) # [B, N, F, d_msg]
        if known_mask is not None:
            m = m * known_mask.unsqueeze(-1).unsqueeze(-1)
        m_cell = torch.sum(alpha.unsqueeze(-1) * m, dim=1) # [B, F, d_msg]
        
        # 4. Drug-Cell Fusion: Condition the drug on the pooled cell sink
        h_fused = self.cell_drug_fusion(torch.cat([h0, m_cell], dim=-1)) # [B, F, d_model]
        
        return self.ln(h0 + h_fused)

class CellSinkGNN(nn.Module):
    def __init__(self, d_model: int, d_msg: int, d_kg: int, num_layers: int, dropout: float = 0.3):
        super().__init__()
        # Stacked layers for multi-hop refinement
        self.layers = nn.ModuleList([
            _CellSinkLayer(d_model, d_msg, d_kg, dropout=dropout) 
            for _ in range(num_layers)
        ])

    def forward(self, h0: Tensor, h_known: Tensor, h_kg: Tensor, known_mask: Tensor | None = None) -> Tensor:
        num_factors = h_kg.shape[2]
        # Expand initial drug embedding for parallel MoE factor routing
        h0 = h0.unsqueeze(1).expand(-1, num_factors, -1)
        
        for layer in self.layers:
            h0 = layer(h0, h_known, h_kg, known_mask=known_mask)
            
        return h0