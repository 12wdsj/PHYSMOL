"""Theory-of-mind scaffold for modelling other agents.

This is a transparent symbolic layer over the current VSA/physics stack.  It
tracks what another agent believes, wants, intends, and feels, including the
possibility that those beliefs differ from reality.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import re


@dataclass
class AgentMind:
    agent_id: str
    beliefs: Dict[str, str] = field(default_factory=dict)
    intentions: List[str] = field(default_factory=list)
    desires: List[str] = field(default_factory=list)
    emotions: Dict[str, float] = field(default_factory=dict)
    observations: List[str] = field(default_factory=list)


class TheoryOfMindModel:
    """Maintain lightweight mental-state models for named agents."""

    def __init__(self):
        self.agents: Dict[str, AgentMind] = {}
        self._emotion_words = {
            "happy": "happy",
            "sad": "sad",
            "angry": "angry",
            "afraid": "afraid",
            "fearful": "afraid",
            "curious": "curious",
            "confused": "confused",
            "高兴": "happy",
            "开心": "happy",
            "难过": "sad",
            "生气": "angry",
            "害怕": "afraid",
            "好奇": "curious",
            "困惑": "confused",
        }

    def ensure_agent(self, agent_id: str) -> AgentMind:
        agent_id = self._clean_agent(agent_id)
        if agent_id not in self.agents:
            self.agents[agent_id] = AgentMind(agent_id)
        return self.agents[agent_id]

    def observe(
        self,
        agent_id: str,
        observation: str = "",
        belief: Optional[str] = None,
        intention: Optional[str] = None,
        desire: Optional[str] = None,
        emotion: Optional[str] = None,
        confidence: float = 0.7,
    ) -> dict:
        mind = self.ensure_agent(agent_id)
        if observation:
            mind.observations.append(observation)
        if belief:
            mind.beliefs[belief] = f"confidence={confidence:.2f}"
        if intention and intention not in mind.intentions:
            mind.intentions.append(intention)
        if desire and desire not in mind.desires:
            mind.desires.append(desire)
        if emotion:
            mind.emotions[emotion] = max(mind.emotions.get(emotion, 0.0), confidence)

        return {
            "kind": "theory_of_mind",
            "agent": mind.agent_id,
            "updated": True,
            "beliefs": dict(mind.beliefs),
            "intentions": list(mind.intentions),
            "desires": list(mind.desires),
            "emotions": dict(mind.emotions),
        }

    def update_from_text(self, text: str) -> Optional[dict]:
        """Extract mental-state assertions from English or Chinese text."""
        stripped = text.strip()

        patterns = [
            (r"(?P<agent>[A-Za-z_][\w-]*)\s+(believes|thinks|knows)\s+(that\s+)?(?P<content>.+)", "belief"),
            (r"(?P<agent>[A-Za-z_][\w-]*)\s+(wants|hopes)\s+(to\s+)?(?P<content>.+)", "desire"),
            (r"(?P<agent>[A-Za-z_][\w-]*)\s+(intends|plans)\s+(to\s+)?(?P<content>.+)", "intention"),
            (r"(?P<agent>[\u4e00-\u9fffA-Za-z_][\u4e00-\u9fff\w-]*)\s*(认为|相信|觉得|知道)\s*(?P<content>.+)", "belief"),
            (r"(?P<agent>[\u4e00-\u9fffA-Za-z_][\u4e00-\u9fff\w-]*)\s*(想要|希望)\s*(?P<content>.+)", "desire"),
            (r"(?P<agent>[\u4e00-\u9fffA-Za-z_][\u4e00-\u9fff\w-]*)\s*(打算|计划)\s*(?P<content>.+)", "intention"),
        ]

        for pattern, kind in patterns:
            match = re.search(pattern, stripped, re.IGNORECASE)
            if not match:
                continue
            agent = match.group("agent")
            content = match.group("content").strip(" .?。？")
            if kind == "belief":
                return self.observe(agent, observation=stripped, belief=content)
            if kind == "desire":
                return self.observe(agent, observation=stripped, desire=content)
            if kind == "intention":
                return self.observe(agent, observation=stripped, intention=content)

        # Emotion-only observations: "Alice is sad", "小明很生气".
        for word, canonical in self._emotion_words.items():
            if word.lower() in stripped.lower():
                agent = self._guess_agent(stripped)
                if agent:
                    return self.observe(agent, observation=stripped, emotion=canonical)

        return None

    def can_answer(self, text: str) -> bool:
        lower = text.lower()
        english_cues = [
            "believe", "believes", "think", "thinks", "know", "knows",
            "intend", "intends", "intention", "want", "wants", "feel",
            "feels", "emotion", "emotions",
        ]
        if "false belief" in lower:
            return True
        if any(re.search(rf"\b{re.escape(cue)}\b", lower) for cue in english_cues):
            return True
        return any(cue in text for cue in ["他人", "认为", "相信", "意图", "情绪", "想要", "觉得"])

    def answer(self, text: str) -> dict:
        agent_id = self._guess_agent(text)
        if not agent_id:
            return {
                "kind": "theory_of_mind",
                "agent": "",
                "answer": "I need to know which agent you mean before I can model their perspective.",
            }

        mind = self.ensure_agent(agent_id)
        lower = text.lower()

        if "believe" in lower or "think" in lower or "认为" in text or "相信" in text:
            answer = self._format_items("beliefs", mind.beliefs.keys())
        elif "intend" in lower or "plan" in lower or "意图" in text or "打算" in text:
            answer = self._format_items("intentions", mind.intentions)
        elif "want" in lower or "desire" in lower or "想要" in text or "希望" in text:
            answer = self._format_items("desires", mind.desires)
        elif "feel" in lower or "emotion" in lower or "情绪" in text or "觉得" in text:
            answer = self._format_items("emotions", mind.emotions.keys())
        else:
            answer = self.describe_agent(agent_id)

        return {
            "kind": "theory_of_mind",
            "agent": mind.agent_id,
            "answer": answer,
            "beliefs": dict(mind.beliefs),
            "intentions": list(mind.intentions),
            "desires": list(mind.desires),
            "emotions": dict(mind.emotions),
        }

    def describe_agent(self, agent_id: str) -> str:
        mind = self.ensure_agent(agent_id)
        parts = [
            self._format_items("beliefs", mind.beliefs.keys()),
            self._format_items("intentions", mind.intentions),
            self._format_items("desires", mind.desires),
            self._format_items("emotions", mind.emotions.keys()),
        ]
        return " ".join(part for part in parts if part)

    def _format_items(self, label: str, items) -> str:
        items = list(items)
        if not items:
            return f"No {label} are known yet."
        return f"Known {label}: " + "; ".join(items)

    def _guess_agent(self, text: str) -> str:
        for agent in self.agents:
            if agent.lower() in text.lower():
                return agent

        english = re.search(r"\b([A-Z][a-zA-Z0-9_-]*)\b", text)
        if english:
            candidate = english.group(1)
            if candidate.lower() not in {"what", "why", "how", "does", "did"}:
                return candidate

        chinese = re.search(r"([\u4e00-\u9fff]{1,8})(认为|相信|觉得|知道|想要|希望|打算|计划|的)", text)
        if chinese:
            return chinese.group(1)

        return ""

    def _clean_agent(self, agent_id: str) -> str:
        return agent_id.strip(" ,.:;!?。？")
