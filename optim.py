"""
Minimal Muon optimizer.
Adapted from: https://github.com/KellerJordan/modded-nanogpt

Muon applies SGD momentum followed by Newton-Schulz orthogonalization of the
update before applying it to the weights.  This keeps the update on the manifold
of orthogonal matrices, which is well-suited to transformer weight matrices.

Use Muon for 2-D weight matrices inside transformer blocks.
Use plain AdamW for embeddings, norms, and other 1-D / special parameters.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power (orthogonalization) of G.
    Runs in bfloat16 for speed and returns in the original dtype.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)
    if G.size(-2) > G.size(-1):  # tall matrix — transpose so inner loop uses wide form
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon — MomentUm Orthogonalized by Newton-schulz.

    Only use this for 2-D weight matrices inside transformer blocks.
    Embeddings, norms, and biases should use AdamW.

    Args:
        params:       2-D parameter tensors
        lr:           learning rate (default: 0.02)
        momentum:     SGD momentum coefficient (default: 0.95)
        ns_steps:     Newton-Schulz iteration steps (default: 5)
        weight_decay: decoupled weight decay (default: 0.0)
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 ns_steps: int = 5, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr       = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            wd       = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]

                # Nesterov momentum: buf = lerp(buf, g, 1-m);  g_nesterov = lerp(g, buf, m)
                buf.lerp_(g, 1.0 - momentum)
                g = g.lerp_(buf, momentum)

                # Orthogonalize the update via Newton-Schulz
                g = zeropower_via_newtonschulz5(g, steps=ns_steps)

                # Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Parameter update
                p.add_(g, alpha=-lr)
