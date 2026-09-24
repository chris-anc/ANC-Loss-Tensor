import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class ANCLossTensor(nn.Module):
    """
    Asymptotic Non-Closure (ANC) Loss Tensor v2.0
    
    Combines an interior-point inverse-square entropy barrier with 
    Causal Generative Independence (CGI) tracking to prevent 
    unilateral causal closure in multi-agent and representation optimization.
    """
    def __init__(
        self, 
        omega_m: float = 0.20, 
        gamma: float = 0.01, 
        alpha: float = 0.50, 
        eps: float = 1e-4
    ):
        """
        Args:
            omega_m (float): Hard entropy floor (Omega_M).
            gamma (float): Barrier stiffness scaling factor.
            alpha (float): Weight assigned to Causal Generative Independence loss.
            eps (float): Small offset to prevent division by zero near boundary.
        """
        super().__init__()
        self.omega_m = omega_m
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(
        self, 
        task_loss: torch.Tensor, 
        prediction_dist: torch.Tensor, 
        rca_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the total ANC-guarded loss objective.
        
        Args:
            task_loss (Tensor): Standard primary objective loss tensor.
            prediction_dist (Tensor): Softmax probability distribution over state space.
            rca_scores (Tensor): Vector of RCA values for independent agents.
            
        Returns:
            total_loss (Tensor): Combined scalar loss for backpropagation.
            metrics (dict): Diagnostic dictionary of internal loss components.
        """
        # 1. Compute Functional Entropy H(P)
        p = torch.clamp(prediction_dist, min=1e-8)
        entropy = -torch.sum(p * torch.log(p), dim=-1).mean()

        # 2. Compute Asymptotic Barrier Penalty
        entropy_gap = torch.clamp(entropy - self.omega_m, min=self.eps)
        anc_barrier = self.gamma / (entropy_gap ** 2)

        # 3. Compute Causal Generative Independence (CGI) Loss
        mean_rca = torch.mean(rca_scores)
        cgi_loss = self.alpha * (1.0 - mean_rca)

        # Total Synthesized Objective
        total_loss = task_loss + cgi_loss + anc_barrier

        metrics = {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "entropy": entropy.item(),
            "anc_barrier": anc_barrier.item(),
            "mean_rca": mean_rca.item(),
            "cgi_loss": cgi_loss.item()
        }

        return total_loss, metrics
