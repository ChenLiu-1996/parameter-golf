from typing import Literal
import torch
from einops import rearrange


class DispersionLoss(torch.nn.Module):
    '''
    Variants (exactly as in the table):

      Decorrelation:      \sum_{(m,n), m != n} Cov_{mn}^2, after l2-normalization
      l2_repel:           log E_{i,j}[exp(-D(z_i, z_j) / \tau_l2)], D(z_i, z_j) = pdist(z_i, z_j, p=2)**2
      Angular spread:     log E_{i,j}[exp(-D(z_i, z_j) / \tau_cos)], D(z_i, z_j) = - z_i z_j / (||z_i|| ||z_j||)
      Orthogonalization:  E_{i,j}[max(0, margin - D(z_i, z_j))^2]
      perplexity entropy: \sum_{(i, j)} p_{j|i} log_2 p_{j|i}, p_{j|i} = exp(-||x_i - x_j||^2 / \sigma^2)

    Notes:
      - \tau_l2, \tau_cos and margin are kept as internal constants for simplicity.
    '''
    def __init__(self,
                 variant: Literal["decorrelation", "l2_repel", "angular_spread", "orthogonalization", "perplexity_entropy"] = "angular_spread",
                 tau_l2: float = 1.0,
                 tau_cos: float = 1.0,
                 margin: float = 0.5,  # NOTE: 0.5 angular cosine distance = orthogonal.
                 epsilon: float = 1e-2,
                 max_tokens: int = 512):
        super().__init__()
        variant = variant.lower()
        assert variant in {"decorrelation", "l2_repel", "angular_spread", "orthogonalization", "perplexity_entropy"}
        self.variant = variant
        self.tau_l2 = float(tau_l2)
        self.tau_cos = float(tau_cos)
        self.margin = float(margin)
        self.epsilon = float(epsilon)
        self.max_tokens = max_tokens

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        '''
        z: [B, L, F],
            where B: batch size. L: sequence length. F: feature dimension.
        '''
        if z.dim() != 3:
            raise ValueError(f'DispersionLoss only supports 3D [B, L, F]; got {tuple(z.shape)}.')

        B, L, F = z.shape

        if self.max_tokens > 0 and L > self.max_tokens:
            idx = torch.randperm(L, device=z.device)[:self.max_tokens]
            z = z[:, idx, :]
            B, L, F = z.shape

        if F < 2:
            raise ValueError(f'DispersionLoss expects F >= 2 in [B, L, F]; got {F}.')

        if self.variant == "decorrelation":
            # NOTE: The covariance matrix `Cov` has shape [B, L, L].
            # \sum_{(m,n), m != n} Cov_{mn}^2, after l2-normalization
            z_centered = (z - z.mean(dim=2, keepdim=True)) / z.std(dim=2, keepdim=True)
            Cov = torch.matmul(z_centered, rearrange(z_centered, 'b l f -> b f l')) / (F - 1)
            non_diag = ~torch.eye(L, dtype=torch.bool, device=z.device)
            return Cov.pow(2)[:, non_diag].mean()

        elif self.variant == "l2_repel":
            # NOTE: The distance matrix matrix `D` has shape [B, L, L].
            D = torch.cdist(z, z)**2
            logit = -D / self.tau_l2
            # Norm regularization to prevent blowing up L2 distance too much.
            norm_regularization = (z ** 2).mean() * 1e-1
            # NOTE: log-sum-exp trick for `log(mean(exp(logit)))`, only differ by a constant: -log(logit.size(1))
            constant_diff = torch.log(torch.tensor(L**2))
            return (torch.logsumexp(logit + self.epsilon, dim=(1, 2)) - constant_diff).mean() + norm_regularization

        elif self.variant == "angular_spread":
            # NOTE: The distance matrix matrix `D` has shape [B, L, L].
            z_norm = z / (torch.linalg.norm(z, dim=2, keepdim=True) + self.epsilon)
            cossim = z_norm @ rearrange(z_norm, 'b l f -> b f l')
            # Clamp to avoid -inf gradient at the two extrema.
            cossim_clamped = torch.clamp(cossim, -1 + self.epsilon, 1 - self.epsilon)
            # Clamp gives 0 gradient beyond the boundary. We force same gradient as the boundary instead.
            cossim_clamped = cossim + (cossim_clamped - cossim).detach()
            D = torch.arccos(cossim_clamped) / torch.pi
            logit = -D / self.tau_cos
            # Set diagonal to -inf.
            mask = torch.eye(L, dtype=torch.bool, device=z.device).unsqueeze(0)
            logit = logit.masked_fill(mask, float('-inf'))
            logit = logit.reshape(B, -1)
            # NOTE: log-sum-exp trick for `log(mean(exp(logit)))`, only differ by a constant: -log(logit.size(1))
            constant_diff = torch.log(torch.tensor(L * (L - 1)))
            return (torch.logsumexp(logit + self.epsilon, dim=1) - constant_diff).mean()

        elif self.variant == 'orthogonalization':
            # NOTE: The distance matrix matrix `D` has shape [B, L, L].
            z_norm = z / (torch.linalg.norm(z, dim=2, keepdim=True) + self.epsilon)
            cossim = z_norm @ rearrange(z_norm, 'b l f -> b f l')
            # Clamp to avoid -inf gradient at the two extrema.
            cossim_clamped = torch.clamp(cossim, -1 + self.epsilon, 1 - self.epsilon)
            # Clamp gives 0 gradient beyond the boundary. We force same gradient as the boundary instead.
            cossim_clamped = cossim + (cossim_clamped - cossim).detach()
            D = torch.arccos(cossim) / torch.pi
            non_diag = ~torch.eye(L, dtype=torch.bool, device=z.device)
            diff = torch.clamp(self.margin - D, min=0.0)
            return diff.pow(2)[:, non_diag].mean()

        elif self.variant == "perplexity_entropy":
            # NOTE: The distance matrix matrix `D` has shape [B, L, L].
            D = torch.cdist(z, z)**2
            # Use fixed sigma (bandwidth)
            sigma_sq = self.tau_l2
            logit = -D / sigma_sq
            # Set diagonal to -inf so p_{i|i} = 0 after softmax
            mask = torch.eye(L, dtype=torch.bool, device=z.device).unsqueeze(0)
            logit = logit.masked_fill(mask, float('-inf'))
            # Compute conditional probabilities p_{j|i} using softmax
            P = torch.softmax(logit, dim=2)
            # Compute entropy: - sum_j p_{j|i} log p_{j|i}
            log_P = torch.log2(P + self.epsilon)
            entropy_per_point = -torch.sum(P * log_P, dim=2)  # [B, L]
            # Minimize negative entropy to maximize entropy.
            return -entropy_per_point.mean()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    B, L, F = 2, 48, 64
    z_base = torch.randn(B, L, F, device=device)

    for variant in [
        "decorrelation",
        "l2_repel",
        "angular_spread",
        "orthogonalization",
        "perplexity_entropy",
    ]:
        print(f"\nVariant: {variant}")
        loss_fn = DispersionLoss(variant=variant)
        z = z_base.clone().requires_grad_(True)
        loss = loss_fn(z)
        print(f"Dispersion loss: {loss.item():.6f}")
        loss.backward()
        print(f"Gradient norm: {torch.norm(z.grad).item():.6f}")
