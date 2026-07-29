"""Flow-matching continuous flow for the per-star likelihood.

`zuko.flows.CNF` is a free-form-Jacobian ODE flow. Trained the way zuko
intends -- maximum likelihood through the adaptive solver -- it is far too
slow to fit this dataset. Trained with the conditional flow-matching
objective it costs about as much per step as a discrete flow, because the
loss is a regression on the velocity field and never solves an ODE.

Density evaluation still needs a solve. `log_prob` uses a fixed-step
midpoint integrator with the *exact* Jacobian trace rather than zuko's
adaptive solver: at 6 target dimensions that is ~3x the cost of an NSF
evaluation, and it is deterministic. Determinism is not optional here --
a Hutchinson trace estimate is free but turns the likelihood into a random
function, which breaks the importance weights of the SMC sampler
downstream.

Integration direction follows zuko's convention: the transform maps data
(t = 0) to base noise (t = 1), so the velocity field points from data to
noise and `sample` integrates backwards.
"""

import torch
import zuko


class FlowMatching(zuko.flows.CNF):
    """Continuous flow trained by flow matching, solved at fixed steps.

    Parameters
    ----------
    features, context : int
        Target and context dimensions.
    steps : int
        Midpoint steps for `log_prob` and `sample`. Two velocity
        evaluations per step; `log_prob` additionally costs one batched
        vector-Jacobian product of width `features` per evaluation.
        Validate it against the adaptive solver (`flow(c).log_prob(x)`,
        zuko's own path) before trusting a posterior -- the per-star error
        is multiplied by the number of stars in a stream.
    chunk : int
        Rows per `log_prob` solve. The exact trace holds `features` copies
        of the activations, so an unchunked inference-sized batch (~800k
        rows) peaks near 18 GiB; chunking costs nothing in time.
    kwargs : dict
        Passed to `zuko.flows.CNF` -> `FFJTransform` -> `zuko.nn.MLP`,
        e.g. `hidden_features`, `activation`, `freqs`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        steps: int = 16,
        chunk: int = 262_144,
        **kwargs,
    ):
        super().__init__(features, context, **kwargs)
        self.features = features
        self.steps = steps
        self.chunk = chunk

    # -- velocity field -----------------------------------------------------

    def velocity(self, t, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Velocity dx/dt at time `t` (0 = data, 1 = base noise)."""
        return self.transform.f(t, x, c)

    def _velocity_and_trace(self, t, x, c, basis):
        """Velocity and its exact divergence, both detached.

        The trace is the sum of the diagonal of df/dx, obtained from one
        vector-Jacobian product per basis vector, batched. Detaching means
        `log_prob` is not differentiable w.r.t. its inputs; nothing in the
        pipeline needs that, and keeping the graph over `steps` solver
        steps would dominate the memory.
        """
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            velocity = self.velocity(t, x, c)
            jacobian = torch.autograd.grad(
                velocity, x, basis, is_grads_batched=True)[0]
        return velocity.detach(), torch.einsum('i...i', jacobian).detach()

    # -- training -----------------------------------------------------------

    def loss(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Conditional flow-matching loss on a batch.

        Regresses the velocity field onto the straight path from each data
        point to a noise draw. No ODE solve, so a step costs one forward
        and one backward pass of the MLP.
        """
        t = torch.rand(x.shape[:-1], device=x.device, dtype=x.dtype)
        z = torch.randn_like(x)
        x_t = torch.lerp(x, z, t[..., None])
        return (self.velocity(t, x_t, c) - (z - x)).square().mean()

    # -- density and sampling -----------------------------------------------

    def log_prob(
        self, x: torch.Tensor, c: torch.Tensor, steps: int = None
    ) -> torch.Tensor:
        """log p(x | c), integrating data -> noise at fixed steps."""
        steps = steps or self.steps
        return torch.cat([
            self._log_prob_chunk(x[i:i + self.chunk], c[i:i + self.chunk],
                                 steps)
            for i in range(0, x.shape[0], self.chunk)
        ])

    def _log_prob_chunk(self, x, c, steps):
        basis = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
        basis = basis.expand(*x.shape, -1).movedim(-1, 0)

        dt = 1.0 / steps
        t = torch.zeros((), dtype=x.dtype, device=x.device)
        ladj = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)

        for _ in range(steps):
            k1 = self.velocity(t, x, c)
            k2, trace = self._velocity_and_trace(
                t + dt / 2, x + k1 * (dt / 2), c, basis)
            x = x + k2 * dt
            ladj = ladj + trace * dt
            t = t + dt

        return self.base().log_prob(x) + ladj

    @torch.no_grad()
    def sample(
        self, c: torch.Tensor, num_samples: int = 1, steps: int = None
    ) -> torch.Tensor:
        """Sample from p(x | c), integrating noise -> data.

        Returns (n_rows, num_samples, features). Cheaper than the discrete
        flows' sampling, which is autoregressive and costs one pass per
        target dimension.
        """
        steps = steps or self.steps
        c = c.unsqueeze(-2).expand(*c.shape[:-1], num_samples, c.shape[-1])
        x = torch.randn(
            *c.shape[:-1], self.features, dtype=c.dtype, device=c.device)

        dt = -1.0 / steps
        t = torch.ones((), dtype=x.dtype, device=x.device)
        for _ in range(steps):
            k1 = self.velocity(t, x, c)
            k2 = self.velocity(t + dt / 2, x + k1 * (dt / 2), c)
            x = x + k2 * dt
            t = t + dt

        return x
