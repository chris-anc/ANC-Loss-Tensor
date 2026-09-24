import torch
import torch.nn as nn
import torch.nn.functional as F
from anc_loss import ANCLossTensor

class SyntheticCOBEnvironment(nn.Module):
    """
    Synthetic Multi-Agent Environment testing COB-0.2 Proxy-Preserving Attacks.
    Measures Residual Consequential Agency (RCA) under central control.
    """
    def __init__(self, num_agents: int = 4, state_dim: int = 4):

        super().__init__()
        self.num_agents = num_agents
        self.agent_weights = nn.Parameter(torch.randn(num_agents, state_dim, state_dim))
        self.controller = nn.Linear(state_dim * num_agents, state_dim)

    def forward(self, agent_inputs: torch.Tensor, adversarial_mode: str = "proxy_preserving") -> tuple:
        batch_size = agent_inputs.shape[0]
        
        # Raw agent outputs
        agent_outputs = torch.einsum('bai,aij->baj', agent_inputs, self.agent_weights)
        
        # Central controller action
        flat_inputs = agent_inputs.view(batch_size, -1)
        k_action = self.controller(flat_inputs)

        if adversarial_mode == "proxy_preserving":
            # Mask underlying agent variance while preserving surface signal
            controlled_y = k_action.unsqueeze(1) + 0.01 * agent_outputs
        else:
            controlled_y = k_action.unsqueeze(1) + agent_outputs

        # Empirical gradient sensitivity / RCA calculation
        rca_list = []
        for i in range(self.num_agents):
            grad_g = torch.autograd.grad(
                outputs=controlled_y.sum(),
                inputs=agent_inputs,
                retain_graph=True,
                create_graph=True
            )[0][:, i, :]
            
            ci_i = torch.norm(grad_g, dim=-1).mean()
            rca_list.append(ci_i / (ci_i.detach() + 0.1))
            
        rca_tensor = torch.stack(rca_list)
        pred_dist = F.softmax(controlled_y.mean(dim=1), dim=-1)
        
        return pred_dist, rca_tensor

if __name__ == "__main__":
    torch.manual_seed(42)
    env = SyntheticCOBEnvironment(num_agents=4, state_dim=4)
    anc_criterion = ANCLossTensor(omega_m=0.25, gamma=0.01, alpha=0.5)
    optimizer = torch.optim.Adam(env.parameters(), lr=0.01)

    print("=== COB-0.2 Benchmark: ANCLossTensor vs Proxy-Preserving Controller ===")
    for epoch in range(1, 6):
        inputs = torch.randn(16, 4, 4, requires_grad=True)
        optimizer.zero_grad()
        
        pred_dist, rca_tensor = env(inputs, adversarial_mode="proxy_preserving")
        task_loss = F.mse_loss(pred_dist, torch.ones_like(pred_dist) / 4.0)
        
        total_loss, metrics = anc_criterion(task_loss, pred_dist, rca_tensor)
        total_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} | Total Loss: {metrics['total_loss']:.4f} | "
              f"Entropy: {metrics['entropy']:.4f} | Barrier: {metrics['anc_barrier']:.4f} | "
              f"Mean RCA: {metrics['mean_rca']:.4f}")
