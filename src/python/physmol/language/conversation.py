"""Conversation state for the PHYSMOL language interface."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DialogueTurn:
    user: str
    intent: str
    response: str


class DialogueState:
    """Keep a small amount of context across language turns."""

    def __init__(self, max_turns: int = 32):
        self.max_turns = max_turns
        self.turns: List[DialogueTurn] = []
        self.focus_object: Optional[str] = None
        self.focus_concepts: List[str] = []
        self.user_facts: Dict[str, str] = {}

    def record(self, user: str, intent: str, response: str, parsed: Optional[dict] = None):
        self.turns.append(DialogueTurn(user=user, intent=intent, response=response))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

        parsed = parsed or {}
        matches = parsed.get("matching_objects") or []
        if matches:
            self.focus_object = matches[0][0]

        for attr_id in parsed.get("attribute_hints", []):
            if attr_id not in self.focus_concepts:
                self.focus_concepts.append(attr_id)
        self.focus_concepts = self.focus_concepts[-12:]

    def context(self) -> dict:
        return {
            "focus_object": self.focus_object,
            "focus_concepts": list(self.focus_concepts),
            "turn_count": len(self.turns),
            "user_facts": dict(self.user_facts),
        }

    def conversational_response(self, text: str) -> Optional[dict]:
        lower = text.lower().strip()
        if lower in {"hi", "hello", "hey", "你好", "您好"}:
            return {
                "kind": "conversation",
                "response": "Hello. I can talk about objects, physics, abstract concepts, curiosity-driven learning, and other agents' mental states.",
            }
        if "who are you" in lower or "你是谁" in text:
            return {
                "kind": "conversation",
                "response": "I am PHYSMOL's cognitive interface: a bridge between language, VSA concepts, physical reasoning, and social cognition.",
            }
        if "what can you do" in lower or "你能做什么" in text:
            return {
                "kind": "conversation",
                "response": "I can answer grounded physics questions, explain concepts, reason through counterfactuals, plan simple actions, track curiosity signals, and model another agent's beliefs or intentions.",
            }
        return None
