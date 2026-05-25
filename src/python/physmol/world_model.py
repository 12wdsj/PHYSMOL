"""PHYSMOL Hierarchical World Model (L1-L2-L3)."""

import numpy as np
from typing import Optional, Dict, Any
from enum import Enum

from .motivation import IntrinsicMotivationSystem


class DecisionLevel(Enum):
    PREDICTOR = 1  # L1: fast reflex, ms timescale
    SIMULATOR = 2  # L2: mental experiments, seconds
    EVOLVER = 3    # L3: structural updates, minutes-hours


class CuriositySignal:
    """Quantifies prediction uncertainty to trigger higher decision levels."""

    def __init__(self, threshold_l2: float = 0.1, threshold_l3: float = 0.5):
        self.threshold_l2 = threshold_l2
        self.threshold_l3 = threshold_l3
        self.history = []

    def compute_error(self, predicted: np.ndarray, observed: np.ndarray) -> float:
        """Prediction error (MSE)."""
        return float(np.mean((predicted - observed) ** 2))

    def should_activate_l2(self, error: float) -> bool:
        return error > self.threshold_l2

    def should_activate_l3(self, recent_errors: list) -> bool:
        """L3 activates when L2 repeatedly fails."""
        if len(recent_errors) < 3:
            return False
        return all(e > self.threshold_l3 for e in recent_errors[-3:])


class PredictorL1:
    """L1: Fast, reflexive single-step prediction."""

    def __init__(self, state_dim: int):
        self.state_dim = state_dim
        # Simple linear model for fast prediction
        self.W = np.eye(state_dim, dtype=np.float32) * 0.99
        self.b = np.zeros(state_dim, dtype=np.float32)

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Single-step state prediction."""
        return state @ self.W + self.b

    def update(self, state: np.ndarray, next_state: np.ndarray, lr: float = 0.01):
        """Online linear regression update."""
        pred = self.predict(state)
        error = pred - next_state
        self.W -= lr * np.outer(state, error)
        self.b -= lr * error


class SimulatorL2:
    """L2: Multi-step forward simulation using causal graph."""

    def __init__(self, state_dim: int, max_steps: int = 10):
        self.state_dim = state_dim
        self.max_steps = max_steps
        self.simulation_log = []

    def simulate(self, state: np.ndarray, actions: list,
                 predictor: PredictorL1) -> list:
        """Run mental experiment: simulate trajectory under given actions."""
        trajectory = [state.copy()]
        current = state.copy()

        for action in actions:
            # Apply action effect (simplified: additive)
            current = current + action
            current = predictor.predict(current)
            trajectory.append(current.copy())

        self.simulation_log.append(trajectory)
        return trajectory

    def evaluate_trajectories(self, trajectories: list) -> int:
        """Select best trajectory (lowest final prediction error)."""
        best_idx = 0
        best_score = float('inf')
        for i, traj in enumerate(trajectories):
            # Score = negative of terminal state energy (lower is better)
            score = np.sum(traj[-1] ** 2)
            if score < best_score:
                best_score = score
                best_idx = i
        return best_idx


class EvolverL3:
    """L3: Structural knowledge updates."""

    def __init__(self):
        self.update_log = []

    def update_vsa_recipe(self, codebook, object_name: str,
                          new_properties: Dict[str, np.ndarray]):
        """Update an object's VSA recipe (structural change)."""
        self.update_log.append({
            'type': 'vsa_recipe_update',
            'object': object_name,
            'properties': list(new_properties.keys())
        })

    def recalibrate_lnn(self, lnn_params, new_data: list):
        """Recalibrate LNN physical constants after repeated L2 failure."""
        self.update_log.append({
            'type': 'lnn_recalibration',
            'data_points': len(new_data)
        })


class HierarchicalWorldModel:
    """Three-tier decision architecture: L1 Predictor / L2 Simulator / L3 Evolver."""

    def __init__(self, state_dim: int = 12):
        self.state_dim = state_dim
        self.l1 = PredictorL1(state_dim)
        self.l2 = SimulatorL2(state_dim)
        self.l3 = EvolverL3()
        self.curiosity = CuriositySignal()
        self.motivation = IntrinsicMotivationSystem()
        self.recent_errors = []
        self.active_level = DecisionLevel.PREDICTOR

    def step(self, state: np.ndarray, action: Optional[np.ndarray] = None,
             observed_next: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Main decision step. Routes through appropriate level."""

        # L1: always predict
        if action is not None:
            state_with_action = state + action
        else:
            state_with_action = state

        prediction = self.l1.predict(state_with_action)

        result = {
            'level': DecisionLevel.PREDICTOR,
            'prediction': prediction,
        }

        if observed_next is None:
            return result

        # Compute error
        error = self.curiosity.compute_error(prediction, observed_next)
        self.recent_errors.append(error)
        uncertainty = float(np.var(prediction - observed_next))
        intrinsic = self.motivation.record_outcome(
            context_key="world_model",
            prediction_error=error,
            uncertainty=uncertainty,
        )
        result['error'] = error
        result['intrinsic_reward'] = intrinsic['intrinsic_reward']
        result['curiosity_components'] = intrinsic

        # Update L1
        self.l1.update(state_with_action, observed_next)

        # Check if L2 should activate
        if self.curiosity.should_activate_l2(error):
            self.active_level = DecisionLevel.SIMULATOR
            result['level'] = DecisionLevel.SIMULATOR
            result['error'] = error

            # L2: run mental experiments with random actions
            candidate_actions = [np.random.randn(self.state_dim) * 0.1
                                 for _ in range(5)]
            trajectories = []
            for act in candidate_actions:
                traj = self.l2.simulate(state, [act], self.l1)
                trajectories.append(traj)

            best = self.l2.evaluate_trajectories(trajectories)
            curiosity_action, action_scores = self.motivation.select_action(
                candidate_actions, context_key="world_model")
            result['best_action'] = candidate_actions[best]
            result['curiosity_action'] = curiosity_action
            result['action_scores'] = action_scores
            result['simulated_trajectory'] = trajectories[best]

            # Check if L3 should activate
            if self.curiosity.should_activate_l3(self.recent_errors):
                self.active_level = DecisionLevel.EVOLVER
                result['level'] = DecisionLevel.EVOLVER
                result['trigger'] = 'repeated_l2_failure'

        return result

    def get_status(self) -> Dict[str, Any]:
        return {
            'active_level': self.active_level,
            'recent_errors': self.recent_errors[-10:],
            'l3_updates': len(self.l3.update_log),
            'motivation': self.motivation.summary(),
        }
