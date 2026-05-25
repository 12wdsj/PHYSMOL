"""Abstract concept reasoning for normative and social concepts.

This module is intentionally small and explicit.  It does not pretend to be a
general moral reasoner; it provides a transparent scaffold that can later be
replaced by learned rule induction from experience and language alignment.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class ConceptRule:
    """A directed rule in the abstract concept graph."""

    source: str
    target: str
    relation: str
    statement: str
    confidence: float = 0.8


@dataclass
class AbstractInference:
    """Result returned by AbstractConceptReasoner.infer."""

    root_concepts: List[str]
    activated_concepts: List[str]
    inferences: List[str]
    applications: List[str]
    chains: List[List[str]]

    def as_dict(self) -> dict:
        return {
            "kind": "abstract_reasoning",
            "root_concepts": self.root_concepts,
            "activated_concepts": self.activated_concepts,
            "inferences": self.inferences,
            "applications": self.applications,
            "chains": self.chains,
        }


class AbstractConceptReasoner:
    """Rule-based bridge from abstract concepts to concrete implications.

    The important addition is multi-hop conceptual grounding.  A concept such
    as "fairness" can activate "justice", which can then activate concrete
    institutional implications such as impartial law enforcement and
    proportionate punishment.
    """

    def __init__(self):
        self.aliases = self._init_aliases()
        self.rules = self._init_rules()
        self.applications = self._init_applications()

        self._by_source: Dict[str, List[ConceptRule]] = {}
        for rule in self.rules:
            self._by_source.setdefault(rule.source, []).append(rule)

    def _init_aliases(self) -> Dict[str, str]:
        return {
            "fairness": "fairness",
            "fair": "fairness",
            "equitable": "fairness",
            "公平": "fairness",
            "justice": "justice",
            "just": "justice",
            "正义": "justice",
            "law": "law",
            "legal": "law",
            "laws": "law",
            "法律": "law",
            "punishment": "punishment",
            "punish": "punishment",
            "penalty": "punishment",
            "惩罚": "punishment",
            "处罚": "punishment",
            "crime": "crime",
            "criminal": "crime",
            "犯罪": "crime",
            "犯罪者": "crime",
            "democracy": "democracy",
            "democratic": "democracy",
            "民主": "democracy",
            "freedom": "freedom",
            "free": "freedom",
            "liberty": "freedom",
            "自由": "freedom",
            "equality": "equality",
            "equal": "equality",
            "平等": "equality",
            "rights": "rights",
            "right": "rights",
            "权利": "rights",
            "slavery": "slavery",
            "slave": "slavery",
            "奴隶制": "slavery",
            "奴役": "slavery",
            "wrong": "wrongness",
            "incorrect": "wrongness",
            "错误": "wrongness",
            "错的": "wrongness",
        }

    def _init_rules(self) -> List[ConceptRule]:
        return [
            ConceptRule(
                "fairness",
                "justice",
                "supports",
                "Fairness requires similar cases to be treated alike unless a relevant difference justifies different treatment.",
                0.9,
            ),
            ConceptRule(
                "fairness",
                "equality",
                "supports",
                "Fairness gives each person comparable moral standing rather than arbitrary privilege.",
                0.85,
            ),
            ConceptRule(
                "justice",
                "law",
                "constrains",
                "If a legal system aims at justice, laws should be applied impartially and with due process.",
                0.9,
            ),
            ConceptRule(
                "justice",
                "punishment",
                "constrains",
                "Justice permits punishment for wrongdoing only when it is evidence-based, proportionate, and not arbitrary.",
                0.88,
            ),
            ConceptRule(
                "crime",
                "punishment",
                "can_trigger",
                "A crime can justify a penalty, but the penalty should be constrained by evidence, proportionality, and rights.",
                0.84,
            ),
            ConceptRule(
                "law",
                "rights",
                "protects",
                "Laws should protect basic rights and provide remedies when rights are violated.",
                0.82,
            ),
            ConceptRule(
                "democracy",
                "equality",
                "requires",
                "Democracy assumes citizens have equal political standing.",
                0.9,
            ),
            ConceptRule(
                "democracy",
                "freedom",
                "requires",
                "Democracy depends on people being free enough to form views, speak, and participate.",
                0.9,
            ),
            ConceptRule(
                "freedom",
                "slavery",
                "conflicts_with",
                "Slavery denies a person's agency and control over their own life.",
                0.95,
            ),
            ConceptRule(
                "equality",
                "slavery",
                "conflicts_with",
                "Slavery creates a hierarchy where some people are treated as property rather than equal persons.",
                0.95,
            ),
            ConceptRule(
                "slavery",
                "wrongness",
                "implies",
                "If people are free and equal, slavery is wrong because it violates both freedom and equality.",
                0.94,
            ),
        ]

    def _init_applications(self) -> Dict[str, List[str]]:
        return {
            "justice": [
                "Law enforcement should aim at justice, so it should avoid arbitrary treatment and keep procedures impartial.",
                "For a proven crime, punishment should be proportionate rather than merely revengeful.",
            ],
            "law": [
                "A law is not only a command; it should be evaluated by whether it protects rights and applies consistently.",
            ],
            "punishment": [
                "Punishment needs evidence, responsibility, and proportionality; otherwise it becomes injustice.",
            ],
            "democracy": [
                "If all citizens are free and politically equal, institutions such as slavery contradict democracy's own premise.",
            ],
            "freedom": [
                "Restrictions on a person's agency require strong justification; total ownership of a person cannot be justified by freedom.",
            ],
            "slavery": [
                "Slavery should be rejected because it removes freedom, denies equal standing, and treats people as property.",
            ],
            "wrongness": [
                "The system can turn an abstract judgment into a policy constraint: do not endorse or plan actions that depend on slavery or arbitrary domination.",
            ],
        }

    def has_signal(self, text: str) -> bool:
        return bool(self.extract_concepts(text))

    def extract_concepts(self, text: str) -> List[str]:
        """Extract canonical abstract concepts from English or Chinese text."""
        text_lower = text.lower()
        found: List[str] = []

        # Phrase matching handles Chinese text without relying on word spaces.
        for alias, concept in sorted(self.aliases.items(), key=lambda x: len(x[0]), reverse=True):
            if alias in text_lower and concept not in found:
                found.append(concept)

        return found

    def infer(self, text: str, max_depth: int = 4) -> AbstractInference:
        roots = self.extract_concepts(text)
        if not roots:
            return AbstractInference([], [], [], [], [])

        activated: Set[str] = set(roots)
        inferences: List[str] = []
        chains: List[List[str]] = []
        frontier: List[Tuple[str, List[str], int]] = [(root, [root], 0) for root in roots]

        while frontier:
            concept, chain, depth = frontier.pop(0)
            if depth >= max_depth:
                continue

            for rule in self._by_source.get(concept, []):
                if rule.statement not in inferences:
                    inferences.append(rule.statement)
                new_chain = chain + [rule.target]
                chains.append(new_chain)

                if rule.target not in activated:
                    activated.add(rule.target)
                    frontier.append((rule.target, new_chain, depth + 1))

        applications = self._collect_applications(activated)
        return AbstractInference(
            root_concepts=roots,
            activated_concepts=sorted(activated),
            inferences=inferences,
            applications=applications,
            chains=chains,
        )

    def explain_concept(self, concept: str) -> dict:
        canonical = self.aliases.get(concept.lower(), concept.lower())
        inference = self.infer(canonical)
        return inference.as_dict()

    def _collect_applications(self, concepts: Iterable[str]) -> List[str]:
        applications: List[str] = []
        for concept in concepts:
            for item in self.applications.get(concept, []):
                if item not in applications:
                    applications.append(item)
        return applications

    def summarize_chain(self, chain: Sequence[str]) -> str:
        return " -> ".join(chain)
