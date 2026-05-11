"""Training and evaluation scaffolds with no-leakage assertions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch import nn

from pipeline.losses import total_loss


@dataclass
class TrainEvalConfig:
    lambda_div: float = 1e-3
    lambda_decorr: float = 0.0 # From updated losses.py
    lambda_sp: float = 1e-3
    warmup_epochs: int = 0
    label_smoothing: float = 0.0


def assert_no_leakage(logits: Tensor, num_classes: int = 188) -> None:
    """Basic structural leakage audit.

    Enforces that model outputs full-candidate logits [B, C].
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected logits rank 2 [B, C], got shape={tuple(logits.shape)}")
    if logits.shape[1] != num_classes:
        raise ValueError(
            f"Expected full-candidate logits with C={num_classes}, got C={logits.shape[1]}"
        )


def train_step(
    model: nn.Module,
    batch: dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
    cfg: TrainEvalConfig,
    epoch: int = 1,
) -> dict[str, float]:
    """Single optimizer step for MoE StarFactor model.

    Expected batch keys:
        h0, h_known, y_known, y
    Optional batch keys:
        h_kg_multi (Stage 2 context)
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    h_kg_multi = batch.get("h_kg_multi")

    # Forward pass utilizing MoE parallel factors
    logits, aux_data = model(
        h0=batch["h0"],
        h_known=batch["h_known"],
        y_known=batch["y_known"],
        h_target=batch["h_target"],
        known_mask=batch.get("known_mask"),
        h_kg_multi=h_kg_multi,
    )

    assert_no_leakage(logits)

    # Warmup scheduling for penalties
    warmup_active = bool(cfg.warmup_epochs > 0 and epoch <= cfg.warmup_epochs)
    effective_lambda_div = 0.0 if warmup_active else float(cfg.lambda_div)
    effective_lambda_decorr = 0.0 if warmup_active else float(cfg.lambda_decorr)
    effective_lambda_sp = 0.0 if warmup_active else float(cfg.lambda_sp)

    # Loss computation with factor diversity and MoE sparsity
    loss, metrics = total_loss(
        logits=logits,
        labels=batch["y"],
        gate_weights=aux_data["gate_weights"],
        factor_weights=aux_data["factor_weights"],
        label_smoothing=cfg.label_smoothing,
        lambda_div=effective_lambda_div,
        lambda_decorr=effective_lambda_decorr,
        lambda_sp=effective_lambda_sp,
    )
    loss.backward()

    # Gradient Tracking for stability
    grad_sq = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_sq += float(torch.sum(param.grad.detach() ** 2).item())
        
    metrics["grad/global_norm"] = float(grad_sq**0.5)
    metrics["train/warmup_active"] = float(warmup_active)
    metrics["train/lambda_div_eff"] = effective_lambda_div
    metrics["train/lambda_decorr_eff"] = effective_lambda_decorr
    metrics["train/lambda_sp_eff"] = effective_lambda_sp

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    return metrics