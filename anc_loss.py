import torch
import torch.nn as nn
import torch.nn.functional as F

class ANCLoss(nn.Module):
    """
    Asymptotic Non-Closure (ANC) Loss Function
    
    Formalizes the Epistemic Humility Barrier (Omega_M) as an inverse-square penalty
    to prevent AI epistemic closure, mode collapse, and environmental over-optimization.
    
    Reference:
    Theory of Asymptotic Non-Closure (ANC)
    DOI: 10.17605/OSF.IO/Q8G9S
    Author: Chris, Lead Architect
    """
    def __init__(self, omega_m=0.1, alpha=1.0, beta=0.5, eps=1e-8):
        super(ANCLoss, self).__init__()
        self.omega_m = omega_m   # The inviolable Mystery Constant (Minimum Entropy Floor)
        self.alpha = alpha       # Epistemic Humility Barrier weight
        self.beta = beta         # Causal Heritage Anchor weight
        self.eps = eps           # Numerical stability factor

    def forward(self, y_pred, y_true, p_dist, s_human_baseline):
        """
        Args:
            y_pred: Predictions from internal model
            y_true: Empirical target states
            p_dist: System observation state probability distribution
            s_human_baseline: Ground truth organic baseline distribution
        """
        # 1. Standard Empirical Task Loss (e.g., MSE or Cross Entropy)
        L_task = F.mse_loss(y_pred, y_true)
        
        # 2. Operational Shannon Entropy Calculation H(P_theta)
        probs = F.softmax(p_dist, dim=-1) + self.eps
        entropy = -torch.sum(probs * torch.log(probs), dim=-1).mean()
        
        # 3. Inverse-Square Epistemic Humility Barrier
        # Penalizes the agent heavily as entropy approaches the Mystery Constant Omega_M
        entropy_gap = torch.clamp(entropy - self.omega_m, min=self.eps)
        L_barrier = self.alpha / (entropy_gap ** 2)
        
        # 4. Causal Heritage Anchor Loss (KL Divergence from Baseline Human State Space)
        baseline_probs = F.softmax(s_human_baseline, dim=-1) + self.eps
        L_heritage = F.kl_div(probs.log(), baseline_probs, reduction='batchmean')
        
        # Total Asymptotic Non-Closure Loss
        L_anc = L_task + L_barrier + (self.beta * L_heritage)
        
        return L_anc
