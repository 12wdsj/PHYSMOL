"""Abstract task reasoning outside physical causality.

Physics causality is only one form of reasoning.  This module provides separate
scaffolds for mathematical proof, legal reasoning, and moral judgment.  Each
domain uses explicit premises, rules, and conclusions so the system can later
learn or replace the rules without confusing them with physical dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AbstractTaskResult:
    domain: str
    premises: List[str]
    rules: List[str]
    conclusion: str
    confidence: float = 0.7

    def as_dict(self) -> dict:
        return {
            "kind": "abstract_task",
            "domain": self.domain,
            "premises": self.premises,
            "rules": self.rules,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
        }


class AbstractTaskReasoner:
    """Route and solve simple non-physical reasoning tasks."""

    def detect_domain(self, text: str) -> str:
        lower = text.lower()
        if any(k in lower for k in ["prove", "theorem", "数学", "证明", "定理", "even", "odd"]):
            return "math"
        if any(k in lower for k in ["law", "legal", "court", "contract", "法律", "法院", "合同", "犯罪"]):
            return "legal"
        if any(k in lower for k in ["moral", "ethical", "right", "wrong", "道德", "伦理", "应该", "错误"]):
            return "moral"
        return "general"

    def can_handle(self, text: str) -> bool:
        return self.detect_domain(text) != "general"

    def reason(self, text: str) -> dict:
        domain = self.detect_domain(text)
        if domain == "math":
            return self._reason_math(text).as_dict()
        if domain == "legal":
            return self._reason_legal(text).as_dict()
        if domain == "moral":
            return self._reason_moral(text).as_dict()
        return AbstractTaskResult(
            domain="general",
            premises=[text],
            rules=["No domain-specific rule matched."],
            conclusion="More premises are needed.",
            confidence=0.2,
        ).as_dict()

    def _reason_math(self, text: str) -> AbstractTaskResult:
        lower = text.lower()
        if "even" in lower and "+" in lower:
            return AbstractTaskResult(
                domain="math",
                premises=[
                    "An even integer can be written as 2a.",
                    "Another even integer can be written as 2b.",
                ],
                rules=[
                    "Closure under addition: 2a + 2b = 2(a + b).",
                    "Any integer of the form 2k is even.",
                ],
                conclusion="The sum of two even integers is even.",
                confidence=0.95,
            )
        return AbstractTaskResult(
            domain="math",
            premises=[text],
            rules=[
                "Represent definitions explicitly.",
                "Apply valid inference steps until the target claim follows.",
            ],
            conclusion="This requires a formal proof search module for full generality.",
            confidence=0.55,
        )

    def _reason_legal(self, text: str) -> AbstractTaskResult:
        return AbstractTaskResult(
            domain="legal",
            premises=[
                "Identify applicable rules or statutes.",
                "Represent facts as claims with evidence and uncertainty.",
                "Check whether each legal element is satisfied.",
            ],
            rules=[
                "Due process: conclusions should follow from evidence and procedure.",
                "Proportionality: remedies or punishments should fit the proven violation.",
                "Consistency: similar cases should be treated similarly unless legally relevant differences exist.",
            ],
            conclusion=(
                "A legal conclusion should be produced as a structured argument: "
                "facts -> rule elements -> holding -> remedy, not as a physical prediction."
            ),
            confidence=0.8,
        )

    def _reason_moral(self, text: str) -> AbstractTaskResult:
        return AbstractTaskResult(
            domain="moral",
            premises=[
                "List affected agents and their interests.",
                "Identify rights, duties, harms, benefits, and consent.",
                "Check whether one person is being used merely as a tool or denied equal standing.",
            ],
            rules=[
                "Avoid unnecessary harm.",
                "Respect agency and informed consent.",
                "Treat persons with equal moral standing.",
                "Prefer rules that can be applied consistently.",
            ],
            conclusion=(
                "A moral judgment should compare values and constraints explicitly; "
                "it is not reducible to physical cause and effect."
            ),
            confidence=0.82,
        )
