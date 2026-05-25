"""PHYSMOL Lagrangian Graph Neural Network (LGNN).

Models physical systems as graphs where:
  - Nodes = particles or rigid bodies
  - Edges = physical interactions (springs, gravity, constraints)

The Lagrangian decomposes as:
  L = Σ_i T_i(q̇_i, m_i) - Σ_{(i,j)∈E} V_ij(q_i, q_j)

Accelerations are derived via the Euler-Lagrange equation using autodiff:
  q̈ = (∇²_{q̇} L)⁻¹ [∇_q L - (∇_q ∇_{q̇} L) q̇]

Reference: Hwang et al., "Learning the Dynamics of Particle-based Systems
with Lagrangian Graph Neural Networks", 2023.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Dict

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Neural network modules (PyTorch)
# ---------------------------------------------------------------------------

if _HAS_TORCH:

    class MLP(nn.Module):
        """Simple multi-layer perceptron with SiLU activation."""

        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                     num_layers: int = 3):
            super().__init__()
            layers = []
            d = in_dim
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(d, hidden_dim))
                layers.append(nn.SiLU())
                d = hidden_dim
            layers.append(nn.Linear(d, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class NodeEnergyNetwork(nn.Module):
        """Per-node kinetic energy network.

        Input:  node state [q_i, q̇_i]  (shape: [N, 2*d_coord])
        Output: kinetic energy contribution per node (shape: [N, 1])
        """

        def __init__(self, coord_dim: int = 2, hidden_dim: int = 64,
                     num_layers: int = 3):
            super().__init__()
            self.net = MLP(2 * coord_dim, hidden_dim, 1, num_layers)

        def forward(self, q: torch.Tensor, q_dot: torch.Tensor) -> torch.Tensor:
            """Args: q, q_dot each [N, d_coord]. Returns [N, 1]."""
            x = torch.cat([q, q_dot], dim=-1)
            return self.net(x)

    class EdgeEnergyNetwork(nn.Module):
        """Per-edge potential energy network.

        Input:  edge features [Δq_ij, ||Δq_ij||]  (shape: [E, d_coord + 1])
        Output: potential energy contribution per edge (shape: [E, 1])
        """

        def __init__(self, coord_dim: int = 2, hidden_dim: int = 64,
                     num_layers: int = 3):
            super().__init__()
            self.net = MLP(coord_dim + 1, hidden_dim, 1, num_layers)

        def forward(self, delta_q: torch.Tensor,
                    dist: torch.Tensor) -> torch.Tensor:
            """Args: delta_q [E, d_coord], dist [E, 1]. Returns [E, 1]."""
            x = torch.cat([delta_q, dist], dim=-1)
            return self.net(x)

else:

    class NodeEnergyNetwork:  # type: ignore[no-redef]
        """Placeholder used when PyTorch is unavailable."""

        pass

    class EdgeEnergyNetwork:  # type: ignore[no-redef]
        """Placeholder used when PyTorch is unavailable."""

        pass


# ---------------------------------------------------------------------------
# Graph data structure (numpy, framework-agnostic)
# ---------------------------------------------------------------------------

class PhysicsGraph:
    """Graph representation of a physical system.

    Attributes:
        num_nodes:     number of particles/bodies
        coord_dim:     dimensionality of each node's coordinates
        edge_index:    (2, E) array — edge_index[0]=sources, edge_index[1]=targets
        edge_type:     (E,) int array — optional edge type for heterogeneous graphs
        node_mass:     (N,) mass per node (used as physical prior)
    """

    def __init__(self, num_nodes: int, coord_dim: int = 2,
                 edge_index: Optional[np.ndarray] = None,
                 edge_type: Optional[np.ndarray] = None,
                 node_mass: Optional[np.ndarray] = None):
        self.num_nodes = num_nodes
        self.coord_dim = coord_dim
        self.edge_index = edge_index  # (2, E)
        self.edge_type = edge_type    # (E,)
        self.node_mass = node_mass if node_mass is not None else np.ones(num_nodes, dtype=np.float32)

    @property
    def num_edges(self) -> int:
        if self.edge_index is None:
            return 0
        return self.edge_index.shape[1]

    def add_edge(self, src: int, dst: int, edge_type: int = 0):
        """Add a single edge (bidirectional by convention)."""
        new_col = np.array([[src, dst], [dst, src]], dtype=np.int64).T
        new_type = np.array([edge_type, edge_type], dtype=np.int64)
        if self.edge_index is None:
            self.edge_index = new_col
            self.edge_type = new_type
        else:
            self.edge_index = np.concatenate([self.edge_index, new_col], axis=1)
            self.edge_type = np.concatenate([self.edge_type, new_type])

    @staticmethod
    def make_chain(num_nodes: int, coord_dim: int = 2,
                   node_mass: Optional[np.ndarray] = None) -> 'PhysicsGraph':
        """Create a linear chain graph (1-2-3-...-N)."""
        g = PhysicsGraph(num_nodes, coord_dim, node_mass=node_mass)
        for i in range(num_nodes - 1):
            g.add_edge(i, i + 1)
        return g

    @staticmethod
    def make_fully_connected(num_nodes: int, coord_dim: int = 2,
                             node_mass: Optional[np.ndarray] = None) -> 'PhysicsGraph':
        """Create a fully connected graph."""
        g = PhysicsGraph(num_nodes, coord_dim, node_mass=node_mass)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                g.add_edge(i, j)
        return g


# ---------------------------------------------------------------------------
# Lagrangian Graph Neural Network
# ---------------------------------------------------------------------------

class LagrangianGraphNetwork:
    """Lagrangian Graph Neural Network.

    Decomposes the Lagrangian into per-node kinetic and per-edge potential
    contributions, then derives accelerations via the Euler-Lagrange equation.

    Unlike the plain LNN (which works on flat q, q̇ vectors), LGNN:
    - Supports variable numbers of nodes and edges
    - Shares parameters across nodes and edges (weight sharing)
    - Can generalize across different topological configurations
    """

    def __init__(self, coord_dim: int = 2, hidden_dim: int = 64,
                 num_layers: int = 3, seed: int = 42):
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for LGNN. Install with: pip install torch"
            )

        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim

        torch.manual_seed(seed)

        self.node_net = NodeEnergyNetwork(coord_dim, hidden_dim, num_layers)
        self.edge_net = EdgeEnergyNetwork(coord_dim, hidden_dim, num_layers)

        self._device = torch.device('cpu')

    def to(self, device: str):
        """Move networks to device ('cpu' or 'cuda')."""
        self._device = torch.device(device)
        self.node_net.to(self._device)
        self.edge_net.to(self._device)
        return self

    def parameters(self):
        """Return all trainable parameters."""
        return list(self.node_net.parameters()) + list(self.edge_net.parameters())

    # ------------------------------------------------------------------
    # Lagrangian computation
    # ------------------------------------------------------------------

    def compute_lagrangian(self, q: torch.Tensor, q_dot: torch.Tensor,
                           graph: PhysicsGraph) -> torch.Tensor:
        """Compute the decomposed Lagrangian L = Σ T_i - Σ V_ij.

        Args:
            q:      (N, d) generalized positions
            q_dot:  (N, d) generalized velocities
            graph:  PhysicsGraph defining topology

        Returns:
            scalar Lagrangian value
        """
        N = graph.num_nodes

        # --- Kinetic energy: sum over nodes ---
        T = self.node_net(q, q_dot).sum()  # scalar

        # --- Potential energy: sum over edges ---
        V = torch.tensor(0.0, dtype=torch.float32, device=self._device)
        if graph.num_edges > 0 and graph.edge_index is not None:
            ei = torch.tensor(graph.edge_index, dtype=torch.long,
                              device=self._device)
            src, dst = ei[0], ei[1]

            delta_q = q[dst] - q[src]              # (E, d)
            dist = torch.norm(delta_q, dim=-1, keepdim=True)  # (E, 1)
            V = self.edge_net(delta_q, dist).sum()  # scalar

        return T - V

    # ------------------------------------------------------------------
    # Euler-Lagrange acceleration
    # ------------------------------------------------------------------

    def compute_acceleration(self, q_np: np.ndarray, q_dot_np: np.ndarray,
                             graph: PhysicsGraph) -> np.ndarray:
        """Compute q̈ via the Euler-Lagrange equation.

        q̈ = (∇²_{q̇} L)⁻¹ [∇_q L - (∇_q ∇_{q̇} L) q̇]

        Args:
            q_np:     (N, d) numpy positions
            q_dot_np: (N, d) numpy velocities
            graph:    PhysicsGraph

        Returns:
            q_ddot: (N, d) numpy accelerations
        """
        N = graph.num_nodes
        d = self.coord_dim
        flat_dim = N * d

        q = torch.tensor(q_np, dtype=torch.float32,
                         device=self._device, requires_grad=True)
        q_dot = torch.tensor(q_dot_np, dtype=torch.float32,
                             device=self._device, requires_grad=True)

        L = self.compute_lagrangian(q, q_dot, graph)

        # ∂L/∂q
        grad_q = torch.autograd.grad(L, q, create_graph=True)[0]  # (N, d)

        # ∂L/∂q̇
        grad_q_dot = torch.autograd.grad(L, q_dot, create_graph=True)[0]  # (N, d)

        # Flatten for Hessian computation
        grad_q_dot_flat = grad_q_dot.reshape(-1)  # (N*d,)
        q_dot_flat = q_dot.reshape(-1)

        # Build Hessian H = ∂²L/∂q̇²  via row-by-row autograd
        H_rows = []
        for i in range(flat_dim):
            row = torch.autograd.grad(
                grad_q_dot_flat[i], q_dot,
                retain_graph=True, create_graph=True
            )[0]  # (N, d)
            H_rows.append(row.reshape(-1))
        H = torch.stack(H_rows, dim=0)  # (N*d, N*d)

        # RHS: ∂L/∂q - (∂²L/∂q∂q̇) q̇
        # Compute ∂²L/∂q∂q̇ via grad of ∂L/∂q w.r.t. q̇
        grad_q_flat = grad_q.reshape(-1)
        cross_rows = []
        for i in range(flat_dim):
            row = torch.autograd.grad(
                grad_q_flat[i], q_dot,
                retain_graph=True, create_graph=True
            )[0]
            cross_rows.append(row.reshape(-1))
        H_cross = torch.stack(cross_rows, dim=0)  # (N*d, N*d)

        rhs = grad_q_flat - H_cross @ q_dot_flat  # (N*d,)

        # Solve H * q̈ = rhs
        # Add small regularization for numerical stability
        H_reg = H + 1e-6 * torch.eye(flat_dim, device=self._device)
        q_ddot_flat = torch.linalg.solve(H_reg, rhs)

        return q_ddot_flat.detach().cpu().numpy().reshape(N, d)

    # ------------------------------------------------------------------
    # Trajectory simulation
    # ------------------------------------------------------------------

    def simulate_trajectory(self, q0: np.ndarray, q_dot0: np.ndarray,
                            graph: PhysicsGraph, dt: float = 0.01,
                            steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate forward trajectory using Euler-Lagrange dynamics.

        Uses Semi-Implicit Euler integration:
          q̇_{t+1} = q̇_t + q̈_t * dt
          q_{t+1} = q_t + q̇_{t+1} * dt

        Args:
            q0:      (N, d) initial positions
            q_dot0:  (N, d) initial velocities
            graph:   PhysicsGraph
            dt:      timestep
            steps:   number of integration steps

        Returns:
            q_traj:     (steps+1, N, d) position trajectory
            q_dot_traj: (steps+1, N, d) velocity trajectory
        """
        N, d = q0.shape
        q_traj = np.zeros((steps + 1, N, d), dtype=np.float32)
        q_dot_traj = np.zeros((steps + 1, N, d), dtype=np.float32)

        q_traj[0] = q0
        q_dot_traj[0] = q_dot0

        for t in range(steps):
            q_ddot = self.compute_acceleration(q_traj[t], q_dot_traj[t], graph)
            q_dot_traj[t + 1] = q_dot_traj[t] + q_ddot * dt
            q_traj[t + 1] = q_traj[t] + q_dot_traj[t + 1] * dt

        return q_traj, q_dot_traj

    # ------------------------------------------------------------------
    # Energy computation
    # ------------------------------------------------------------------

    def energy(self, q_np: np.ndarray, q_dot_np: np.ndarray,
               graph: PhysicsGraph) -> float:
        """Compute total energy T + V.

        Since L = T - V, we need separate T and V.
        """
        q = torch.tensor(q_np, dtype=torch.float32, device=self._device)
        q_dot = torch.tensor(q_dot_np, dtype=torch.float32, device=self._device)

        with torch.no_grad():
            T = self.node_net(q, q_dot).sum().item()

            V = 0.0
            if graph.num_edges > 0 and graph.edge_index is not None:
                ei = torch.tensor(graph.edge_index, dtype=torch.long,
                                  device=self._device)
                src, dst = ei[0], ei[1]
                delta_q = q[dst] - q[src]
                dist = torch.norm(delta_q, dim=-1, keepdim=True)
                V = self.edge_net(delta_q, dist).sum().item()

        return T + V

    def lagrangian(self, q_np: np.ndarray, q_dot_np: np.ndarray,
                   graph: PhysicsGraph) -> float:
        """Compute L = T - V."""
        q = torch.tensor(q_np, dtype=torch.float32, device=self._device)
        q_dot = torch.tensor(q_dot_np, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            return self.compute_lagrangian(q, q_dot, graph).item()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            'node_net': self.node_net.state_dict(),
            'edge_net': self.edge_net.state_dict(),
        }

    def load_state_dict(self, state: dict):
        self.node_net.load_state_dict(state['node_net'])
        self.edge_net.load_state_dict(state['edge_net'])

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        state = torch.load(path, map_location=self._device,
                           weights_only=True)
        self.load_state_dict(state)


# ---------------------------------------------------------------------------
# Analytical physics baselines (for testing / ground truth)
# ---------------------------------------------------------------------------

class SpringMassSystem:
    """Analytical two-body spring system for ground-truth comparison.

    Potential:  V = 0.5 * k * (||q1 - q2|| - L0)^2
    Kinetic:    T = 0.5 * m1 * ||q̇1||^2 + 0.5 * m2 * ||q̇2||^2
    """

    def __init__(self, m1: float = 1.0, m2: float = 1.0,
                 k: float = 5.0, L0: float = 1.0):
        self.m1 = m1
        self.m2 = m2
        self.k = k
        self.L0 = L0

    def potential(self, q: np.ndarray) -> float:
        """q: (2, d) positions of the two bodies."""
        delta = q[1] - q[0]
        dist = np.linalg.norm(delta)
        return 0.5 * self.k * (dist - self.L0) ** 2

    def kinetic(self, q_dot: np.ndarray) -> float:
        """q_dot: (2, d) velocities."""
        return 0.5 * self.m1 * np.dot(q_dot[0], q_dot[0]) + \
               0.5 * self.m2 * np.dot(q_dot[1], q_dot[1])

    def energy(self, q: np.ndarray, q_dot: np.ndarray) -> float:
        return self.kinetic(q_dot) + self.potential(q)

    def acceleration(self, q: np.ndarray) -> np.ndarray:
        """Analytical acceleration from spring force.

        F_0 = k * (||q1-q2|| - L0) * (q1-q0)/||q1-q0||
        When stretched (dist > L0): positive force toward body 1 (attractive)
        When compressed (dist < L0): negative force away from body 1 (repulsive)
        """
        delta = q[1] - q[0]
        dist = np.linalg.norm(delta)
        if dist < 1e-8:
            return np.zeros_like(q)
        force_mag = self.k * (dist - self.L0)
        force_dir = delta / dist
        f1 = force_mag * force_dir
        f2 = -f1
        a1 = f1 / self.m1
        a2 = f2 / self.m2
        return np.array([a1, a2], dtype=np.float32)

    def simulate(self, q0: np.ndarray, q_dot0: np.ndarray,
                 dt: float = 0.01, steps: int = 100):
        """Semi-Implicit Euler integration."""
        N, d = q0.shape
        q_traj = np.zeros((steps + 1, N, d), dtype=np.float32)
        q_dot_traj = np.zeros((steps + 1, N, d), dtype=np.float32)
        q_traj[0] = q0
        q_dot_traj[0] = q_dot0

        for t in range(steps):
            a = self.acceleration(q_traj[t])
            q_dot_traj[t + 1] = q_dot_traj[t] + a * dt
            q_traj[t + 1] = q_traj[t] + q_dot_traj[t + 1] * dt

        return q_traj, q_dot_traj


class GravitationalTwoBody:
    """Analytical two-body gravitational system.

    Potential:  V = -G * m1 * m2 / ||q1 - q2||
    Kinetic:    T = 0.5 * m1 * ||q̇1||^2 + 0.5 * m2 * ||q̇2||^2

    Softened: V = -G * m1 * m2 / sqrt(||q1 - q2||^2 + ε^2)
    """

    def __init__(self, m1: float = 1.0, m2: float = 1.0,
                 G: float = 1.0, softening: float = 0.1):
        self.m1 = m1
        self.m2 = m2
        self.G = G
        self.softening = softening

    def potential(self, q: np.ndarray) -> float:
        delta = q[1] - q[0]
        dist = np.sqrt(np.dot(delta, delta) + self.softening ** 2)
        return -self.G * self.m1 * self.m2 / dist

    def kinetic(self, q_dot: np.ndarray) -> float:
        return 0.5 * self.m1 * np.dot(q_dot[0], q_dot[0]) + \
               0.5 * self.m2 * np.dot(q_dot[1], q_dot[1])

    def energy(self, q: np.ndarray, q_dot: np.ndarray) -> float:
        return self.kinetic(q_dot) + self.potential(q)

    def acceleration(self, q: np.ndarray) -> np.ndarray:
        delta = q[1] - q[0]
        dist_sq = np.dot(delta, delta) + self.softening ** 2
        dist = np.sqrt(dist_sq)
        force_mag = self.G * self.m1 * self.m2 / (dist_sq * dist)
        force_dir = delta / dist
        f1 = force_mag * force_dir
        f2 = -f1
        a1 = f1 / self.m1
        a2 = f2 / self.m2
        return np.array([a1, a2], dtype=np.float32)

    def simulate(self, q0: np.ndarray, q_dot0: np.ndarray,
                 dt: float = 0.01, steps: int = 100):
        N, d = q0.shape
        q_traj = np.zeros((steps + 1, N, d), dtype=np.float32)
        q_dot_traj = np.zeros((steps + 1, N, d), dtype=np.float32)
        q_traj[0] = q0
        q_dot_traj[0] = q_dot0

        for t in range(steps):
            a = self.acceleration(q_traj[t])
            q_dot_traj[t + 1] = q_dot_traj[t] + a * dt
            q_traj[t + 1] = q_traj[t] + q_dot_traj[t + 1] * dt

        return q_traj, q_dot_traj
