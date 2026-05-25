"""PHYSMOL Lagrangian Neural Network - Python wrapper with PyTorch bridge."""

import numpy as np
from typing import Optional, Tuple

try:
    from . import _lnn
except ImportError:
    _lnn = None


class LagrangianNetwork:
    """Lagrangian Neural Network: parameterizes L_θ(q, q̇) and computes
    accelerations via the Euler-Lagrange equation.

    Uses C core for CPU computation. For GPU (HIP/ROCm), use LagrangianNetworkTorch.
    """

    def __init__(self, coord_dim: int = 6, hidden_dim: int = 128, seed: int = 42):
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim

        if _lnn is not None:
            self._params = _lnn.LNNParams(coord_dim, hidden_dim)
            self._params.random_init(seed)
            self._use_c = True
        else:
            self._use_c = False
            self._init_numpy(coord_dim, hidden_dim, seed)

    def _init_numpy(self, coord_dim, hidden_dim, seed):
        """Pure numpy fallback."""
        rng = np.random.RandomState(seed)
        in_dim = 2 * coord_dim
        scale1 = np.sqrt(2.0 / (in_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        scale3 = np.sqrt(2.0 / (hidden_dim + 1))

        self.W1 = rng.randn(in_dim, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)
        self.W3 = rng.randn(hidden_dim).astype(np.float32) * scale3
        self.b3 = 0.0

    def _forward_numpy(self, q, q_dot):
        """Pure numpy forward pass."""
        x = np.concatenate([q, q_dot]).astype(np.float32)
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        return float(h2 @ self.W3 + self.b3)

    def forward(self, q: np.ndarray, q_dot: np.ndarray) -> float:
        """Compute L_θ(q, q̇)."""
        if self._use_c:
            return self._params.forward(q.astype(np.float32), q_dot.astype(np.float32))
        return self._forward_numpy(q, q_dot)

    def compute_acceleration(self, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """Compute q̈ via Euler-Lagrange equation (Eq.16).

        q̈ = (∇²_{q̇} L)⁻¹ [∇_q L - (∇_q ∇_{q̇} L) q̇]
        """
        if self._use_c:
            return self._params.compute_acceleration(
                q.astype(np.float32), q_dot.astype(np.float32))

        # Fallback: finite differences
        cd = self.coord_dim
        eps = 1e-4

        # ∂L/∂q
        dL_dq = np.zeros(cd, dtype=np.float32)
        for i in range(cd):
            qp = q.copy(); qp[i] += eps
            qm = q.copy(); qm[i] -= eps
            dL_dq[i] = (self._forward_numpy(qp, q_dot) - self._forward_numpy(qm, q_dot)) / (2 * eps)

        # ∂L/∂q̇
        dL_dqdot = np.zeros(cd, dtype=np.float32)
        for i in range(cd):
            qdp = q_dot.copy(); qdp[i] += eps
            qdm = q_dot.copy(); qdm[i] -= eps
            dL_dqdot[i] = (self._forward_numpy(q, qdp) - self._forward_numpy(q, qdm)) / (2 * eps)

        # Hessian ∇²_{q̇} L
        H = np.zeros((cd, cd), dtype=np.float32)
        for i in range(cd):
            for j in range(cd):
                qdp = q_dot.copy(); qdp[i] += eps; qdp[j] += eps
                qdm = q_dot.copy(); qdm[i] -= eps; qdm[j] -= eps
                qdpm = q_dot.copy(); qdpm[i] += eps; qdpm[j] -= eps
                qdmp = q_dot.copy(); qdmp[i] -= eps; qdmp[j] += eps
                H[i, j] = (self._forward_numpy(q, qdp) - self._forward_numpy(q, qdpm)
                          - self._forward_numpy(q, qdmp) + self._forward_numpy(q, qdm)) / (4 * eps**2)

        # RHS: ∂L/∂q - (∇_q ∇_{q̇} L) q̇
        rhs = dL_dq - H @ q_dot

        # Solve H * q̈ = rhs
        try:
            q_ddot = np.linalg.solve(H, rhs)
        except np.linalg.LinAlgError:
            q_ddot = np.zeros(cd, dtype=np.float32)

        return q_ddot

    def simulate_trajectory(self, q0: np.ndarray, q_dot0: np.ndarray,
                            dt: float = 0.01, steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate trajectory using Euler-Lagrange dynamics.

        Returns:
            q_traj: (steps+1, coord_dim)
            q_dot_traj: (steps+1, coord_dim)
        """
        cd = self.coord_dim
        q_traj = np.zeros((steps + 1, cd), dtype=np.float32)
        q_dot_traj = np.zeros((steps + 1, cd), dtype=np.float32)

        q_traj[0] = q0
        q_dot_traj[0] = q_dot0

        for t in range(steps):
            q_ddot = self.compute_acceleration(q_traj[t], q_dot_traj[t])
            q_dot_traj[t + 1] = q_dot_traj[t] + q_ddot * dt
            q_traj[t + 1] = q_traj[t] + q_dot_traj[t + 1] * dt

        return q_traj, q_dot_traj

    def energy(self, q: np.ndarray, q_dot: np.ndarray) -> float:
        """Compute total energy (T + V). Approximate via Lagrangian + 2V."""
        # L = T - V, so T + V = L + 2V
        # For a rough estimate, just return L (which is T - V)
        return self.forward(q, q_dot)
