"""Loss functions for StarFactor MoE training."""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor

def cross_entropy_loss(logits: Tensor, labels: Tensor, label_smoothing: float = 0.0) -> Tensor:
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

def moe_sparsity_loss(gate_weights: Tensor, eps: float = 1e-8) -> Tensor:
    """Encourages the MoE router to be confident/sparse in its selection."""
    # Entropy minimization: smaller entropy = more confident routing
    entropy = -torch.sum(gate_weights * torch.log(gate_weights + eps), dim=-1)
    return torch.mean(entropy)

def factor_diversity_loss(alpha_rel: Tensor | None) -> Tensor:
    """Penalizes high cosine similarity between factor weight distributions.
    Ensures that different factors look at different KG relations.
    """
    if alpha_rel is None:
        return torch.tensor(0.0, device=torch.device('cpu'))
        
    # alpha_rel: [F, C, R]. We flatten to [F, C * R]
    F_dim = alpha_rel.shape[0]
    flat = alpha_rel.view(F_dim, -1)
    flat = F.normalize(flat, p=2, dim=-1)
    
    penalty = torch.zeros((), device=alpha_rel.device, dtype=alpha_rel.dtype)
    for i in range(F_dim):
        for j in range(i + 1, F_dim):
            # We want dot product to be minimal (orthogonal)
            penalty = penalty + torch.sum(flat[i] * flat[j])
            
    return penalty

def factor_decorrelation_loss(alpha_rel: Tensor | None) -> Tensor:
    """Penalizes statistical correlation between factors.
    Computes the cross-correlation matrix of the factors and minimizes the off-diagonal elements.
    """
    if alpha_rel is None:
        return torch.tensor(0.0, device=torch.device('cpu'))
        
    F_dim = alpha_rel.shape[0]
    if F_dim <= 1:
        return torch.tensor(0.0, device=alpha_rel.device)
        
    flat = alpha_rel.view(F_dim, -1)  # [F, C * R]
    
    # 1. Center the factors by subtracting the mean
    flat_centered = flat - flat.mean(dim=1, keepdim=True)
    
    # 2. Compute the covariance matrix [F, F]
    cov = flat_centered @ flat_centered.t() / (flat.shape[1] - 1)
    
    # 3. Normalize to a correlation matrix (values between -1 and 1)
    var = torch.diag(cov).clamp_min(1e-8)
    std = torch.sqrt(var)
    corr = cov / torch.outer(std, std)
    
    # 4. Extract and penalize the off-diagonal elements
    eye = torch.eye(F_dim, device=alpha_rel.device, dtype=alpha_rel.dtype)
    off_diagonal = corr - eye
    
    # Squared Frobenius norm normalized by the number of off-diagonal elements
    penalty = torch.sum(off_diagonal ** 2) / (F_dim * (F_dim - 1))
    return penalty

def total_loss(
    logits: Tensor,
    labels: Tensor,
    gate_weights: Tensor,
    factor_weights: Tensor | None,
    label_smoothing: float,
    lambda_div: float,
    lambda_decorr: float,
    lambda_sp: float,
) -> tuple[Tensor, dict[str, float]]:

    ce = cross_entropy_loss(logits, labels, label_smoothing=label_smoothing)
    
    sp_loss = moe_sparsity_loss(gate_weights) if lambda_sp > 0 else torch.tensor(0.0, device=logits.device)
    div_loss = factor_diversity_loss(factor_weights) if lambda_div > 0 else torch.tensor(0.0, device=logits.device)
    decorr_loss = factor_decorrelation_loss(factor_weights) if lambda_decorr > 0 else torch.tensor(0.0, device=logits.device)

    loss = ce + (lambda_sp * sp_loss) + (lambda_div * div_loss) + (lambda_decorr * decorr_loss)
    
    metrics = {
        "loss/total": float(loss.detach().cpu()),
        "loss/ce": float(ce.detach().cpu()),
        "loss/moe_sparsity": float(sp_loss.detach().cpu()),
        "loss/factor_diversity": float(div_loss.detach().cpu()),
        "loss/factor_decorrelation": float(decorr_loss.detach().cpu()),
        "train/label_smoothing": float(label_smoothing),
    }
    return loss, metrics