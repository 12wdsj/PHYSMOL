"""PHYSMOL LNN Python Integration Tests."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

from physmol.lnn import LagrangianNetwork


class TestLagrangianNetwork:
    def test_create(self):
        lnn = LagrangianNetwork(coord_dim=3, hidden_dim=64)
        assert lnn.coord_dim == 3
        assert lnn.hidden_dim == 64

    def test_forward(self):
        lnn = LagrangianNetwork(coord_dim=3, hidden_dim=64, seed=42)
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        q_dot = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        L = lnn.forward(q, q_dot)
        assert isinstance(L, float)
        assert np.isfinite(L)

    def test_forward_deterministic(self):
        lnn = LagrangianNetwork(coord_dim=3, hidden_dim=64, seed=42)
        q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        q_dot = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        L1 = lnn.forward(q, q_dot)
        L2 = lnn.forward(q, q_dot)
        assert abs(L1 - L2) < 1e-6

    def test_acceleration(self):
        lnn = LagrangianNetwork(coord_dim=3, hidden_dim=64, seed=42)
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        q_dot = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        q_ddot = lnn.compute_acceleration(q, q_dot)
        assert len(q_ddot) == 3
        assert all(np.isfinite(q_ddot))

    def test_simulate_trajectory(self):
        lnn = LagrangianNetwork(coord_dim=2, hidden_dim=32, seed=42)
        q0 = np.array([1.0, 0.0], dtype=np.float32)
        q_dot0 = np.array([0.0, 0.0], dtype=np.float32)
        q_traj, q_dot_traj = lnn.simulate_trajectory(q0, q_dot0, dt=0.01, steps=50)
        assert q_traj.shape == (51, 2)
        assert q_dot_traj.shape == (51, 2)
        assert all(np.isfinite(q_traj.flatten()))
        assert all(np.isfinite(q_dot_traj.flatten()))

    def test_energy_conservation(self):
        """Energy should be approximately conserved for short trajectories."""
        lnn = LagrangianNetwork(coord_dim=2, hidden_dim=32, seed=42)
        q0 = np.array([1.0, 0.0], dtype=np.float32)
        q_dot0 = np.array([0.5, 0.0], dtype=np.float32)
        q_traj, q_dot_traj = lnn.simulate_trajectory(q0, q_dot0, dt=0.001, steps=100)

        energies = [lnn.energy(q_traj[t], q_dot_traj[t]) for t in range(0, 101, 10)]
        # Energy should not diverge wildly
        e_range = max(energies) - min(energies)
        e_mean = np.mean(np.abs(energies))
        if e_mean > 0:
            assert e_range / e_mean < 10.0  # rough check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
