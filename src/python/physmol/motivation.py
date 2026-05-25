"""Intrinsic motivation and curiosity-driven action scoring.

The goal here is to make curiosity operational rather than just a threshold on
prediction error.  The module tracks novelty, uncertainty, and learning
progress per context, then turns those signals into an intrinsic reward.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MotivationTrace:
    """History for one context or action region."""

    key: str
    visits: int = 0
    prediction_errors: List[float] = field(default_factory=list)
    uncertainty: float = 0.0

    def recent_error(self, window: int = 5) -> float:
        if not self.prediction_errors:
            return 0.0
        return float(np.mean(self.prediction_errors[-window:]))

    def previous_error(self, window: int = 5) -> float:
        if len(self.prediction_errors) <= window:
            return self.recent_error(window)
        start = max(0, len(self.prediction_errors) - 2 * window)
        prev = self.prediction_errors[start:-window]
        if not prev:
            return self.recent_error(window)
        return float(np.mean(prev))

    def learning_progress(self, window: int = 5) -> float:
        """Positive when recent error is lower than the previous window."""
        return max(0.0, self.previous_error(window) - self.recent_error(window))


class IntrinsicMotivationSystem:
    """Curiosity signal based on novelty, uncertainty, and progress."""

    def __init__(
        self,
        novelty_weight: float = 0.35,
        uncertainty_weight: float = 0.25,
        progress_weight: float = 0.40,
    ):
        self.novelty_weight = novelty_weight
        self.uncertainty_weight = uncertainty_weight
        self.progress_weight = progress_weight
        self._traces: Dict[str, MotivationTrace] = {}
        self.last_reward: float = 0.0
        self.last_components: Dict[str, float] = {}

    def record_outcome(
        self,
        context_key: str,
        prediction_error: float,
        uncertainty: Optional[float] = None,
    ) -> Dict[str, float]:
        """Record a prediction outcome and return intrinsic reward terms."""
        trace = self._traces.setdefault(context_key, MotivationTrace(context_key))
        trace.visits += 1
        trace.prediction_errors.append(float(prediction_error))
        if uncertainty is not None:
            trace.uncertainty = float(max(0.0, uncertainty))

        novelty = 1.0 / np.sqrt(trace.visits)
        progress = trace.learning_progress()
        uncertainty_term = trace.uncertainty

        reward = (
            self.novelty_weight * novelty
            + self.uncertainty_weight * uncertainty_term
            + self.progress_weight * progress
        )
        reward = float(max(0.0, reward))

        self.last_reward = reward
        self.last_components = {
            "novelty": float(novelty),
            "uncertainty": float(uncertainty_term),
            "learning_progress": float(progress),
            "intrinsic_reward": reward,
        }
        return dict(self.last_components)

    def score_action(self, action: Any, context_key: str = "default") -> float:
        """Score a candidate action before execution.

        Unknown actions receive a novelty bonus.  Known contexts receive a
        progress bonus when the model is actively improving there.
        """
        action_key = self._action_key(action, context_key)
        trace = self._traces.get(action_key)
        if trace is None:
            return self.novelty_weight

        novelty = 1.0 / np.sqrt(trace.visits + 1)
        return float(
            self.novelty_weight * novelty
            + self.uncertainty_weight * trace.uncertainty
            + self.progress_weight * trace.learning_progress()
        )

    def select_action(
        self,
        candidate_actions: Sequence[Any],
        context_key: str = "default",
    ) -> Tuple[Any, Dict[str, float]]:
        """Select the most intrinsically valuable action."""
        if not candidate_actions:
            raise ValueError("candidate_actions must not be empty")

        scores = {
            str(i): self.score_action(action, context_key)
            for i, action in enumerate(candidate_actions)
        }
        best_idx = max(scores, key=scores.get)
        return candidate_actions[int(best_idx)], scores

    def summary(self) -> Dict[str, Any]:
        return {
            "num_contexts": len(self._traces),
            "last_reward": self.last_reward,
            "last_components": dict(self.last_components),
            "contexts": {
                key: {
                    "visits": trace.visits,
                    "recent_error": trace.recent_error(),
                    "learning_progress": trace.learning_progress(),
                    "uncertainty": trace.uncertainty,
                }
                for key, trace in self._traces.items()
            },
        }

    def _action_key(self, action: Any, context_key: str) -> str:
        if isinstance(action, np.ndarray):
            rounded = tuple(np.round(action.astype(float), 2).tolist())
            return f"{context_key}:action:{rounded}"
        return f"{context_key}:action:{repr(action)}"
