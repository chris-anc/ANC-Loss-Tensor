# Theory of Asymptotic Non-Closure (ANC) v2.0
> **Preventing Unilateral Causal Closure in Optimization and Autonomous Systems**

[![License: MIT](httpsbadge.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.17605%2FOSF.IO%2FQ8G9S-blue.svg)](https://doi.org/10.17605/OSF.IO/Q8G9S)

The **Theory of Asymptotic Non-Closure (ANC)** is an interior-point barrier optimization framework designed to prevent **Premature Unilateral Causal Closure (PUCC)** during gradient descent. By coupling an inverse-square entropy barrier ($\Omega_M$) with continuous differential tracking of **Residual Consequential Agency (RCA)**, ANC restricts optimizers from driving systemic option-spaces or independent agent input pathways below viable operational baselines.

---

## Key Features

- **Interior-Point Barrier ($\Omega_M$):** Imposes an asymptotic penalty wall as state entropy $H(P)$ approaches a hard baseline floor.
- **Causal Generative Independence ($L_{\text{CGI}}$):** Differentiable loss tracking that prevents central models from suppressing independent input gradient pathways.
- **COB Benchmark Resilience:** Validated against **COB-0.2 Proxy-Preserving Attacks**, maintaining agent influence ($\text{RCA} > 0.85$) where standard cross-entropy collapses ($\text{RCA} \approx 0.06$).
- **Generative Independence Renewal (GIR):** Establishes the formal macro-system condition for permanent non-closure ($\tau_{\text{spawn}} < \tau_{\text{annex}}$).

---

## Mathematical Formulation

$$L_{\text{ANC}} = L_{\text{Task}} + L_{\text{CGI}} + \frac{\gamma}{(\max(H(P) - \Omega_M, \epsilon))^2}$$

Where:
- $L_{\text{Task}}$: Primary task/objective loss.
- $H(P)$: Functional entropy of prediction/state distribution $P$.
- $\Omega_M$: Minimum allowable entropy floor (interior-point limit).
- $\gamma$: Barrier stiffness coefficient.
- $L_{\text{CGI}}$: $\alpha \cdot (1 - \overline{\text{RCA}})$, penalizing causal centralization.

---

## Quick Start (PyTorch)

```python
import torch
from anc_loss import ANCLossTensor

# Initialize ANC Loss Tensor v2.0
anc_criterion = ANCLossTensor(
    omega_m=0.20,  # Hard entropy floor
    gamma=0.01,    # Barrier stiffness
    alpha=0.50     # CGI loss weight
)

# In your training loop:
# pred_dist: Softmax probability distribution (batch_size, num_classes)
# rca_scores: Tensor of computed RCA values for independent agents
total_loss, metrics = anc_criterion(task_loss, pred_dist, rca_scores)

total_loss.backward()
