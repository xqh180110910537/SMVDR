import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class GatedMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, top_k=1, tau=1.0):
        super(GatedMoE, self).__init__()
        self.experts = nn.ModuleList([FFN(input_dim, hidden_dim) for _ in range(num_experts)])
        self.Wg = nn.Parameter(torch.randn(input_dim, num_experts))
        self.b = nn.Parameter(torch.randn(num_experts))
        # self.Wnoise = nn.Parameter(torch.randn(input_dim, num_experts))
        self.top_k = top_k
        # self.tau = tau
        self.final_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_list):
        batchsize = x_list[0].size(0)

        expert_outputs = []
        gates = []
        expert_probabilities = []
        for x in x_list:
            # Step 1: Add noise
            H = x @ self.Wg + self.b
            # noise = torch.randn_like(H)  # StandardNormal()
            # H += noise * F.softplus(x @ self.Wnoise)
            expert_probabilities.append(F.softmax(H, dim=-1))

            top_k_values, _ = torch.topk(H, self.top_k, dim=-1)
            mask = H >= top_k_values[:, -1, None]
            H_top_k = H.masked_fill(~mask, float('-inf'))

            # G = F.gumbel_softmax(H_top_k, tau=self.tau, hard=False)
            G = F.softmax(H_top_k, dim=1)
            gates.append(G)
            expert_outputs.append(torch.stack([expert(x) for expert in self.experts], dim=1))

        # Combine expert outputs with gates
        combined_output = sum(torch.sum(G.unsqueeze(-1) * e, dim=1)
                              for G, e in zip(gates, expert_outputs)) / len(x_list)

        output = self.final_fc(combined_output)

        # expert_probabilities = [F.softmax(prob, dim=-1) for prob in expert_probabilities]
        return output, expert_probabilities


def entropy(p):
    return -torch.sum(p * torch.log(p + 1e-9), dim=-1)


def entropy_regularization_loss(expert_probabilities):
    M = len(expert_probabilities)
    batch_size = expert_probabilities[0].size(0)
    H_mj = [entropy(prob).mean() for prob in expert_probabilities]
    avg_prob = torch.stack(expert_probabilities, dim=0).mean(dim=0)
    H_avg = entropy(avg_prob).mean()
    E = torch.abs((1 / M) * sum(H_mj) - H_avg)

    return E
