"""Trainable abstract reasoning components for PHYSMOL.

The first implementation is deliberately lightweight and inspectable:

* ProofRuleLibrary learns reusable proof rules from math examples.
* LegalCaseBase stores cases and retrieves analogous cases by factors.
* ValueConstraintLearner accumulates moral values and action constraints.

These are not neural fine-tuning yet; they are the trainable substrate that can
later supervise a neural responder or symbolic proof/search engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .long_term_memory import LongTermMemory
from .progress import ProgressLogger
from .training_data import TrainingExample


@dataclass
class ProofRule:
    name: str
    domain: str
    pattern: str
    premises: List[str]
    conclusion_template: str
    uses: int = 1

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "domain": self.domain,
            "pattern": self.pattern,
            "premises": self.premises,
            "conclusion_template": self.conclusion_template,
            "uses": self.uses,
        }


@dataclass
class LegalCase:
    case_id: str
    facts: str
    issue: str
    rule: str
    holding: str
    factors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "facts": self.facts,
            "issue": self.issue,
            "rule": self.rule,
            "holding": self.holding,
            "factors": self.factors,
        }


@dataclass
class ValueConstraint:
    value: str
    action_pattern: str
    constraint: str
    polarity: str = "avoid"
    support: int = 1

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "action_pattern": self.action_pattern,
            "constraint": self.constraint,
            "polarity": self.polarity,
            "support": self.support,
        }


class ProofRuleLibrary:
    """Small proof rule learner and matcher."""

    def __init__(self):
        self.rules: Dict[str, ProofRule] = {}
        self._init_defaults()

    def train_example(self, ex: TrainingExample) -> Optional[ProofRule]:
        text = f"{ex.text}\n{ex.target or ''}".lower()
        if "even" in text and ("2k" in text or "2a" in text or "2(" in text):
            return self.add_rule(ProofRule(
                name="even_addition_closure",
                domain="math",
                pattern="even + even",
                premises=[
                    "Represent even integers as multiples of 2.",
                    "Factor out 2 after addition.",
                ],
                conclusion_template="The sum is even because it has the form 2k.",
            ))
        if "contradiction" in text or "反证" in text:
            return self.add_rule(ProofRule(
                name="proof_by_contradiction",
                domain="math",
                pattern="assume not target -> contradiction",
                premises=[
                    "Assume the negation of the desired claim.",
                    "Derive an impossibility from the assumptions.",
                ],
                conclusion_template="The original claim follows because its negation is impossible.",
            ))
        return None

    def add_rule(self, rule: ProofRule) -> ProofRule:
        existing = self.rules.get(rule.name)
        if existing:
            existing.uses += 1
            return existing
        self.rules[rule.name] = rule
        return rule

    def solve(self, query: str) -> dict:
        lower = query.lower()
        matches = [
            rule for rule in self.rules.values()
            if all(part in lower for part in rule.pattern.split() if part != "+")
        ]
        if not matches and "even" in lower:
            matches = [self.rules["even_addition_closure"]]
        if not matches:
            return {
                "kind": "proof_search",
                "query": query,
                "steps": [],
                "conclusion": "No proof rule matched yet.",
            }
        rule = sorted(matches, key=lambda r: r.uses, reverse=True)[0]
        steps = [*rule.premises, rule.conclusion_template]
        return {
            "kind": "proof_search",
            "query": query,
            "rule": rule.name,
            "steps": steps,
            "conclusion": rule.conclusion_template,
        }

    def to_dict(self) -> dict:
        return {"rules": [rule.to_dict() for rule in self.rules.values()]}

    def _init_defaults(self):
        self.add_rule(ProofRule(
            name="even_addition_closure",
            domain="math",
            pattern="even + even",
            premises=[
                "Let the first even integer be 2a.",
                "Let the second even integer be 2b.",
                "Then 2a + 2b = 2(a + b).",
            ],
            conclusion_template="The sum of two even integers is even.",
        ))


class LegalCaseBase:
    """Case-based legal reasoning memory."""

    def __init__(self):
        self.cases: Dict[str, LegalCase] = {}
        self._counter = 0

    def train_example(self, ex: TrainingExample) -> Optional[LegalCase]:
        text = ex.text
        target = ex.target or ""
        if not self._looks_legal(text, target):
            return None
        factors = self.extract_factors(f"{text} {target}")
        self._counter += 1
        case = LegalCase(
            case_id=f"case_{self._counter:06d}",
            facts=text,
            issue=self._extract_issue(text),
            rule=self._extract_rule(target),
            holding=target or "No holding provided.",
            factors=factors,
        )
        self.cases[case.case_id] = case
        return case

    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        q_factors = set(self.extract_factors(query))
        scored = []
        for case in self.cases.values():
            c_factors = set(case.factors)
            overlap = len(q_factors.intersection(c_factors))
            denom = max(1, len(q_factors.union(c_factors)))
            score = overlap / denom
            scored.append((case, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            {**case.to_dict(), "score": score}
            for case, score in scored[:top_k]
        ]

    def extract_factors(self, text: str) -> List[str]:
        lower = text.lower()
        factors = []
        mapping = {
            "evidence": ["evidence", "proof", "证据"],
            "intent": ["intent", "intention", "mens rea", "故意", "意图"],
            "harm": ["harm", "damage", "injury", "伤害", "损害"],
            "contract": ["contract", "agreement", "合同", "协议"],
            "punishment": ["punishment", "penalty", "sentence", "惩罚", "刑罚"],
            "due_process": ["due process", "procedure", "程序", "正当程序"],
            "proportionality": ["proportionate", "proportionality", "比例"],
            "rights": ["rights", "right", "权利"],
        }
        for factor, keywords in mapping.items():
            if any(k in lower or k in text for k in keywords):
                factors.append(factor)
        return factors or ["general_legal_reasoning"]

    def to_dict(self) -> dict:
        return {"cases": [case.to_dict() for case in self.cases.values()]}

    def _looks_legal(self, text: str, target: str) -> bool:
        combined = f"{text} {target}".lower()
        return any(k in combined for k in ["law", "legal", "court", "contract", "crime"]) or \
            any(k in f"{text} {target}" for k in ["法律", "法院", "合同", "犯罪", "惩罚"])

    def _extract_issue(self, text: str) -> str:
        if "?" in text:
            return text.split("?", 1)[0].strip() + "?"
        if "？" in text:
            return text.split("？", 1)[0].strip() + "？"
        return "Determine the legal consequence."

    def _extract_rule(self, target: str) -> str:
        if not target:
            return "Apply relevant legal elements."
        sentences = re.split(r"[。.!?]", target)
        return sentences[0].strip() or "Apply relevant legal elements."


class ValueConstraintLearner:
    """Learn moral values and constraints from examples."""

    def __init__(self):
        self.constraints: Dict[str, ValueConstraint] = {}

    def train_example(self, ex: TrainingExample) -> Optional[ValueConstraint]:
        text = f"{ex.text}\n{ex.target or ''}"
        values = self.extract_values(text)
        if not values:
            return None
        action = self.extract_action_pattern(text)
        value = values[0]
        polarity = "avoid" if self._has_negative_judgment(text) else "prefer"
        constraint = self._make_constraint(value, action, polarity)
        key = f"{value}:{action}:{polarity}"
        existing = self.constraints.get(key)
        if existing:
            existing.support += 1
            return existing
        learned = ValueConstraint(value, action, constraint, polarity=polarity)
        self.constraints[key] = learned
        return learned

    def judge(self, action_description: str) -> dict:
        values = self.extract_values(action_description)
        matches = []
        lower = action_description.lower()
        for item in self.constraints.values():
            if item.value in values or item.action_pattern in lower:
                matches.append(item.to_dict())
        return {
            "kind": "value_constraint_judgment",
            "action": action_description,
            "matched_constraints": matches,
            "judgment": self._summarize(matches),
        }

    def extract_values(self, text: str) -> List[str]:
        lower = text.lower()
        values = []
        mapping = {
            "freedom": ["freedom", "liberty", "自由"],
            "equality": ["equality", "equal", "平等"],
            "harm": ["harm", "suffering", "伤害", "痛苦"],
            "consent": ["consent", "voluntary", "同意", "自愿"],
            "justice": ["justice", "fair", "正义", "公平"],
            "rights": ["rights", "权利"],
        }
        for value, keywords in mapping.items():
            if any(k in lower or k in text for k in keywords):
                values.append(value)
        return values

    def extract_action_pattern(self, text: str) -> str:
        lower = text.lower()
        if "slavery" in lower or "奴隶" in text:
            return "enslave_or_dominate_person"
        if "punish" in lower or "惩罚" in text:
            return "punish_person"
        if "lie" in lower or "欺骗" in text:
            return "deceive_person"
        if "harm" in lower or "伤害" in text:
            return "harm_person"
        return "general_action"

    def to_dict(self) -> dict:
        return {"constraints": [item.to_dict() for item in self.constraints.values()]}

    def _has_negative_judgment(self, text: str) -> bool:
        lower = text.lower()
        return any(k in lower for k in ["wrong", "bad", "unjust", "immoral", "avoid"]) or \
            any(k in text for k in ["错误", "不正义", "不道德", "避免"])

    def _make_constraint(self, value: str, action: str, polarity: str) -> str:
        verb = "Avoid" if polarity == "avoid" else "Prefer"
        return f"{verb} actions matching {action} when they violate {value}."

    def _summarize(self, matches: Sequence[dict]) -> str:
        if not matches:
            return "No learned moral constraint matched yet."
        avoid = [m for m in matches if m.get("polarity") == "avoid"]
        if avoid:
            return "The action is constrained by learned avoid-rules: " + "; ".join(m["constraint"] for m in avoid)
        return "The action is supported by learned prefer-rules: " + "; ".join(m["constraint"] for m in matches)


class AbstractCognitionTrainer:
    """Train proof, legal, and moral abstract components."""

    def __init__(self, memory: Optional[LongTermMemory] = None):
        self.memory = memory or LongTermMemory()
        self.proofs = ProofRuleLibrary()
        self.legal_cases = LegalCaseBase()
        self.values = ValueConstraintLearner()

    def train(
        self,
        examples: Iterable[TrainingExample],
        progress: Optional[ProgressLogger] = None,
        total_steps: Optional[int] = None,
    ) -> dict:
        examples = list(examples)
        total = total_steps or len(examples)
        metrics = {"proof_rules": 0, "legal_cases": 0, "value_constraints": 0, "memories": 0}

        for idx, ex in enumerate(examples, start=1):
            proof_rule = self.proofs.train_example(ex)
            legal_case = self.legal_cases.train_example(ex)
            value_constraint = self.values.train_example(ex)

            if proof_rule:
                metrics["proof_rules"] = len(self.proofs.rules)
            if legal_case:
                metrics["legal_cases"] = len(self.legal_cases.cases)
            if value_constraint:
                metrics["value_constraints"] = len(self.values.constraints)

            self.memory.add_episode(
                content=ex.text,
                outcome=ex.target,
                tags=["abstract_training", ex.metadata.get("task", ex.modality)],
            )
            metrics["memories"] += 1

            if progress:
                progress.log(
                    step=idx,
                    total_steps=total,
                    phase="abstract_training",
                    metrics=dict(metrics),
                    message=f"Processed {idx}/{total}: {ex.text[:80]}",
                )

        return metrics

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "proof_rules.json"), "w", encoding="utf-8") as f:
            json.dump(self.proofs.to_dict(), f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "legal_cases.json"), "w", encoding="utf-8") as f:
            json.dump(self.legal_cases.to_dict(), f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "value_constraints.json"), "w", encoding="utf-8") as f:
            json.dump(self.values.to_dict(), f, ensure_ascii=False, indent=2)
        self.memory.save_json(os.path.join(out_dir, "abstract_memory.json"))
