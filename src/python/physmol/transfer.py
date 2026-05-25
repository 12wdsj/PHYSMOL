"""Cross-domain transfer through abstract relational schemas.

LGNN can generalize across physical topologies, but it cannot by itself move
knowledge from blocks to chess or from physics to law.  This module provides a
separate transfer layer: concrete experiences are lifted into schemas, then
schemas are mapped into a target domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass
class Domain:
    name: str
    primitives: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)


@dataclass
class TransferSchema:
    name: str
    abstraction: str
    source_patterns: List[str]
    target_patterns: List[str]
    confidence: float = 0.7


class CrossDomainTransferEngine:
    """Map source-domain patterns to target-domain hypotheses."""

    def __init__(self):
        self.domains: Dict[str, Domain] = {}
        self.schemas: List[TransferSchema] = []
        self._init_defaults()

    def register_domain(
        self,
        name: str,
        primitives: Optional[Sequence[str]] = None,
        relations: Optional[Sequence[str]] = None,
        goals: Optional[Sequence[str]] = None,
    ):
        self.domains[name] = Domain(
            name=name,
            primitives=list(primitives or []),
            relations=list(relations or []),
            goals=list(goals or []),
        )

    def add_schema(self, schema: TransferSchema):
        self.schemas.append(schema)

    def propose_transfer(
        self,
        source_domain: str,
        target_domain: str,
        observed_patterns: Sequence[str],
        top_k: int = 5,
    ) -> dict:
        """Return target-domain hypotheses triggered by observed patterns."""
        observed = {p.lower() for p in observed_patterns}
        candidates = []
        for schema in self.schemas:
            source_overlap = [
                pattern for pattern in schema.source_patterns
                if pattern.lower() in observed
            ]
            if not source_overlap:
                continue

            target_hits = [
                pattern for pattern in schema.target_patterns
                if self._domain_supports(target_domain, pattern)
            ] or schema.target_patterns

            score = schema.confidence * len(source_overlap) / len(schema.source_patterns)
            candidates.append({
                "schema": schema.name,
                "abstraction": schema.abstraction,
                "source_domain": source_domain,
                "target_domain": target_domain,
                "matched_source_patterns": source_overlap,
                "target_hypotheses": target_hits,
                "score": score,
            })

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return {
            "kind": "cross_domain_transfer",
            "source_domain": source_domain,
            "target_domain": target_domain,
            "hypotheses": candidates[:top_k],
        }

    def lift_experience_to_patterns(self, text: str) -> List[str]:
        """Heuristic pattern extraction for early bootstrapping."""
        lower = text.lower()
        patterns = []
        keyword_map = {
            "block": "spatial_support",
            "积木": "spatial_support",
            "stack": "support_structure",
            "堆叠": "support_structure",
            "fall": "stability_failure",
            "倒塌": "stability_failure",
            "push": "action_changes_state",
            "推动": "action_changes_state",
            "path": "path_constraint",
            "路径": "path_constraint",
            "obstacle": "blocking_constraint",
            "阻挡": "blocking_constraint",
            "collision": "interaction_changes_motion",
            "碰撞": "interaction_changes_motion",
            "chess": "turn_based_planning",
            "下棋": "turn_based_planning",
            "棋": "turn_based_planning",
            "move": "action_changes_state",
            "走子": "action_changes_state",
            "attack": "threat_relation",
            "攻击": "threat_relation",
            "defend": "support_structure",
            "防守": "support_structure",
            "check": "goal_constraint",
            "将军": "goal_constraint",
        }
        for keyword, pattern in keyword_map.items():
            if keyword in lower and pattern not in patterns:
                patterns.append(pattern)
        return patterns

    def _domain_supports(self, domain_name: str, pattern: str) -> bool:
        domain = self.domains.get(domain_name)
        if domain is None:
            return True
        values = set(domain.primitives + domain.relations + domain.goals)
        return pattern in values or not values

    def _init_defaults(self):
        self.register_domain(
            "blocks",
            primitives=["block", "surface", "tower", "gap"],
            relations=[
                "spatial_support", "support_structure", "stability_failure",
                "blocking_constraint", "path_constraint", "action_changes_state",
            ],
            goals=["stable_stack", "move_object"],
        )
        self.register_domain(
            "chess",
            primitives=["piece", "square", "king", "move"],
            relations=[
                "support_structure", "blocking_constraint", "path_constraint",
                "turn_based_planning", "threat_relation", "goal_constraint",
                "action_changes_state",
            ],
            goals=["checkmate", "material_gain", "king_safety"],
        )

        self.schemas.extend([
            TransferSchema(
                name="constraint_transfer",
                abstraction="A relation can limit which actions are legal or useful.",
                source_patterns=["blocking_constraint", "path_constraint"],
                target_patterns=["blocking_constraint", "path_constraint"],
                confidence=0.85,
            ),
            TransferSchema(
                name="support_transfer",
                abstraction="Entities can protect or stabilize other entities.",
                source_patterns=["spatial_support", "support_structure"],
                target_patterns=["support_structure", "king_safety"],
                confidence=0.78,
            ),
            TransferSchema(
                name="planning_transfer",
                abstraction="A useful action is evaluated by future state changes, not only immediate motion.",
                source_patterns=["action_changes_state", "stability_failure"],
                target_patterns=["turn_based_planning", "goal_constraint", "action_changes_state"],
                confidence=0.82,
            ),
            TransferSchema(
                name="threat_transfer",
                abstraction="An interaction can create risk for another entity and force defensive action.",
                source_patterns=["interaction_changes_motion", "blocking_constraint"],
                target_patterns=["threat_relation", "king_safety"],
                confidence=0.72,
            ),
        ])
