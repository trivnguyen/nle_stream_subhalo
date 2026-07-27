
import torch
import numpy as np
from scipy.stats import truncnorm

from .basic import StarBatch

class UncertaintySampler:
    """
    Sample a per-star measurement sigma for one feature, add Gaussian noise
    to that feature, and append the sigma as a column of `batch.xerr` so the
    NLE model can condition on it. Only perturbed features get a column.
    """
    def __init__(self, distribution_type, feature_idx, **params):
        """
        Args:
            distribution_type: distribution to sample the sigma from
                ('uniform', 'gaussian', 'jeffreys', 'gamma',
                'uniform_varied', 'jeffreys_varied').
            feature_idx: Column of `batch.x` to perturb.
            **params: Parameters for the distribution
                For 'uniform': low, high (range for uniform sampling of std)
                For 'gaussian': mean, std (parameters for Gaussian sampling of std)
        """
        self.distribution_type = distribution_type
        self.feature_idx = feature_idx
        self.params = params

        # Validate parameters
        if distribution_type == 'uniform':
            if 'low' not in params or 'high' not in params:
                raise ValueError("Uniform distribution requires 'low' and 'high' parameters")
            if params['low'] < 0 or params['high'] < 0:
                raise ValueError("Uncertainty parameters must be non-negative")
            if params['low'] > params['high']:
                raise ValueError("'low' must be <= 'high' for uniform distribution")
        elif distribution_type == 'gaussian':
            if 'mean' not in params or 'std' not in params:
                raise ValueError("Gaussian distribution requires 'mean' and 'std' parameters")
            if params['mean'] < 0 or params['std'] < 0:
                raise ValueError("Uncertainty parameters must be non-negative")
        elif distribution_type == 'jeffreys':
            if 'low' not in params or 'high' not in params:
                raise ValueError("Jeffreys prior requires 'low' and 'high' parameters")
            if params['low'] <= 0 or params['high'] <= 0:
                raise ValueError("'low' and 'high' must be positive for Jeffreys prior")
            if params['low'] > params['high']:
                raise ValueError("'low' must be <= 'high' for Jeffreys prior")
        elif distribution_type == 'gamma':
            if 'alpha' not in params or 'beta' not in params or 'x0' not in params:
                raise ValueError("Gamma distribution requires 'alpha', 'beta', and 'x0' parameters")
            if params['alpha'] < -1:
                raise ValueError("'alpha' must be > -1")
            if params['beta'] <= 0:
                raise ValueError("'beta' must be > 0")
            if params['x0'] <= 0:
                raise ValueError("'x0' must be > 0")
        elif distribution_type == 'uniform_varied':
            if 'low_range' not in params or 'width_range' not in params:
                raise ValueError("Uniform distribution requires 'low_range' and 'width_range' parameters")
            if params['low'][0] < 0 or params['low'][1] < 0:
                raise ValueError("`low_range` values must be non-negative")
            if params['width_range'][0] <= 0 or params['width_range'][1] <= 0:
                raise ValueError("`width_range` values must be positive")
            if params['low'][0] > params['low'][1]:
                raise ValueError("'low_range[0]' must be <= 'low_range[1]'")
            if params['width_range'][0] > params['width_range'][1]:
                raise ValueError("'width_range[0]' must be <= 'width_range[1]'")
        elif distribution_type == 'jeffreys_varied':
            if 'low_range' not in params or 'width_range' not in params:
                raise ValueError("Uniform distribution requires 'low_range' and 'width_range' parameters")
            if params['low_range'][0] <= 0 or params['low_range'][1] <= 0:
                raise ValueError("`low_range` values must be positive")
            if params['width_range'][0] <= 0 or params['width_range'][1] <= 0:
                raise ValueError("`width_range` values must be positive")
            if params['low_range'][0] > params['low_range'][1]:
                raise ValueError("'low_range[0]' must be <= 'low_range[1]'")
            if params['width_range'][0] > params['width_range'][1]:
                raise ValueError("'width_range[0]' must be <= 'width_range[1]'")
        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")

    def sample_uncertainty_std(self, n_samples=1):
        """
        Sample uncertainty standard deviation values.

        Args:
            n_samples: Number of std values to sample

        Returns:
            std_values: Tensor of sampled standard deviation values (no gradient)
        """
        if self.distribution_type == 'uniform':
            low = self.params['low']
            high = self.params['high']
            std_values = torch.rand(n_samples) * (high - low) + low
        elif self.distribution_type == 'gaussian':
            mean = self.params['mean']
            std = self.params['std']

            # Use truncated Gaussian with lower bound at 0
            # Define truncation bounds in terms of standard deviations from mean
            a = (0 - mean) / std  # Lower bound in standardized form
            b = float('inf')      # Upper bound (no upper limit)

            # Sample from truncated normal
            samples = truncnorm.rvs(a, b, loc=mean, scale=std, size=n_samples)
            std_values = torch.tensor(samples, dtype=torch.float32)
        elif self.distribution_type == 'jeffreys':
            # Jeffreys prior for a Gaussian with known mean: p(sigma) = 1 / sigma
            # This assumes the mean is known and we're only estimating sigma
            low = self.params['low']
            high = self.params['high']
            if low <= 0 or high <= 0:
                raise ValueError("Jeffreys prior requires 'low' and 'high' to be positive")

            # sample log sigma uniformly to get p(sigma) ∝ 1/sigma
            log_low = np.log(low)
            log_high = np.log(high)
            log_std_values = torch.rand(n_samples) * (log_high - log_low) + log_low
            std_values = torch.exp(log_std_values)
        elif self.distribution_type == 'gamma':
            alpha = self.params['alpha']
            beta = self.params['beta']
            x0 = self.params['x0']

            # shape parameter for gamma distribution
            k = (alpha + 1) / beta

            # Sample from Gamma(k, 1) distribution and x = x0 * z^(1 / beta)
            gamma_dist = torch.distributions.Gamma(k, 1)
            z = gamma_dist.sample((n_samples,))
            std_values = x0 * torch.pow(z, 1 / beta)
        elif self.distribution_type == 'uniform_varied':
            low_range = self.params['low_range']
            width_range = self.params['width_range']

            # ensure that high and low are sampled once per batch
            low_samples = torch.rand(1) * (low_range[1] - low_range[0]) + low_range[0]
            width_samples = torch.rand(1) * (width_range[1] - width_range[0]) + width_range[0]
            std_values = low_samples + torch.rand(n_samples) * width_samples
        elif self.distribution_type == 'jeffreys_varied':
            low_range = self.params['low_range']
            width_range = self.params['width_range']

            # ensure that high and low are sampled once per batch
            low_samples = torch.rand(1) * (low_range[1] - low_range[0]) + low_range[0]
            width_samples = torch.rand(1) * (width_range[1] - width_range[0]) + width_range[0]
            log_low_samples = torch.log(low_samples)
            log_high_samples = torch.log(low_samples + width_samples)

            log_std_values = torch.rand(n_samples) * (log_high_samples - log_low_samples) + log_low_samples
            std_values = torch.exp(log_std_values)
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")

        return std_values.detach()

    def __call__(self, batch: StarBatch) -> StarBatch:
        """
        Perturb one feature with Gaussian measurement noise and append its
        sigma to `batch.xerr`.

        Args:
            batch: StarBatch with `batch.x` of shape (B, k).

        Returns:
            batch: Modified StarBatch. `batch.x[:, feature_idx]` is noised
                and the sampled sigma is appended as the last column of
                `batch.xerr`, so `xerr` has one column per sampler (in the
                order the samplers run), not one per feature.
        """
        batch = batch.clone()  # Ensure we don't modify the original batch
        x = batch.x

        if x.dim() != 2:
            raise ValueError(
                f"Expected 2D x [N_batch, k], got {x.dim()}D")
        if self.feature_idx >= x.shape[1] or self.feature_idx < 0:
            raise ValueError(
                f"feature_idx {self.feature_idx} out of range for x with "
                f"{x.shape[1]} components")

        std_values = self.sample_uncertainty_std(x.shape[0]).to(x)
        x[:, self.feature_idx] += torch.normal(0, std_values)

        err_col = std_values.unsqueeze(1)
        batch.xerr = (
            err_col if batch.xerr is None
            else torch.cat([batch.xerr, err_col], dim=1))

        batch.x = x
        return batch

    def get_distribution_info(self):
        """Return information about the current distribution configuration."""
        return {
            'distribution_type': self.distribution_type,
            'parameters': self.params
        }

