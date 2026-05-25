"""Long-term memory for facts, episodes, and accumulated experience.

VSA recipes are good at storing attribute patterns, but they are not enough for
lifelong learning.  This module adds a separate memory layer for:

* episodic memory: "what happened, when, with whom, and what changed"
* semantic facts: stable statements the system can reuse later
* experience traces: action -> observation -> outcome records for training

The memory records can still be indexed with VSA/text vectors, but the content
is not collapsed into VSA recipes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MemoryRecord:
    memory_id: str
    memory_type: str
    content: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    strength: float = 1.0
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "strength": self.strength,
            "timestamp": self.timestamp,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryRecord":
        return cls(
            memory_id=data["memory_id"],
            memory_type=data["memory_type"],
            content=data["content"],
            tags=list(data.get("tags", [])),
            metadata=dict(data.get("metadata", {})),
            strength=float(data.get("strength", 1.0)),
            timestamp=float(data.get("timestamp", time.time())),
            embedding=data.get("embedding"),
        )


class LongTermMemory:
    """Persistent memory store with vector and keyword retrieval."""

    def __init__(self, text_encoder=None, store_embeddings: bool = False):
        self.text_encoder = text_encoder
        self.store_embeddings = store_embeddings
        self._records: Dict[str, MemoryRecord] = {}
        self._counter = 0

    def add_episode(
        self,
        content: str,
        actors: Optional[Sequence[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        outcome: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> MemoryRecord:
        metadata = {"actors": list(actors or []), "context": context or {}}
        if outcome:
            metadata["outcome"] = outcome
        return self._add("episode", content, tags or [], metadata)

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        evidence: Optional[str] = None,
        confidence: float = 0.8,
        tags: Optional[Sequence[str]] = None,
    ) -> MemoryRecord:
        content = f"{subject} {predicate} {obj}"
        metadata = {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": float(confidence),
        }
        if evidence:
            metadata["evidence"] = evidence
        return self._add("fact", content, tags or [], metadata, strength=confidence)

    def add_experience(
        self,
        description: str,
        action: Optional[Any] = None,
        observation: Optional[Any] = None,
        reward: Optional[float] = None,
        prediction_error: Optional[float] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> MemoryRecord:
        metadata = {
            "action": self._compact(action),
            "observation": self._compact(observation),
        }
        if reward is not None:
            metadata["reward"] = float(reward)
        if prediction_error is not None:
            metadata["prediction_error"] = float(prediction_error)
        return self._add("experience", description, tags or [], metadata)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        memory_types: Optional[Iterable[str]] = None,
    ) -> List[Tuple[MemoryRecord, float]]:
        """Retrieve records using vector similarity plus keyword overlap."""
        allowed = set(memory_types) if memory_types else None
        records = [
            rec for rec in self._records.values()
            if allowed is None or rec.memory_type in allowed
        ]
        if not records:
            return []

        q_vec = self._encode(query)
        q_terms = set(self._terms(query))
        scored: List[Tuple[MemoryRecord, float]] = []
        for rec in records:
            keyword_score = self._keyword_score(q_terms, rec)
            vector_score = 0.0
            if q_vec is not None:
                r_vec = self._record_vector(rec)
                if r_vec is not None:
                    denom = np.linalg.norm(q_vec) * np.linalg.norm(r_vec)
                    if denom > 1e-10:
                        vector_score = float(np.dot(q_vec, r_vec) / denom)
            recency = 1.0 / (1.0 + max(0.0, time.time() - rec.timestamp) / 86400.0)
            score = 0.55 * vector_score + 0.35 * keyword_score + 0.10 * recency
            score *= rec.strength
            scored.append((rec, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def consolidate(self, min_repetitions: int = 2) -> List[MemoryRecord]:
        """Promote repeated experience patterns into semantic fact records."""
        buckets: Dict[str, List[MemoryRecord]] = {}
        for rec in self._records.values():
            if rec.memory_type not in {"episode", "experience"}:
                continue
            key = "|".join(sorted(rec.tags)) or rec.content.lower()
            buckets.setdefault(key, []).append(rec)

        created: List[MemoryRecord] = []
        for key, records in buckets.items():
            if len(records) < min_repetitions:
                continue
            fact_text = f"Repeated experience pattern: {key}"
            if any(r.content == fact_text for r in self._records.values()):
                continue
            created.append(self.add_fact(
                subject="experience",
                predicate="suggests",
                obj=key,
                evidence=f"{len(records)} supporting records",
                confidence=min(0.95, 0.5 + 0.1 * len(records)),
                tags=["consolidated", *key.split("|")],
            ))
        return created

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([rec.to_dict() for rec in self._records.values()],
                      f, ensure_ascii=False, indent=2)

    def load_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        self._records = {
            rec.memory_id: rec for rec in
            (MemoryRecord.from_dict(item) for item in records)
        }
        self._counter = len(self._records)

    def list_records(self, memory_type: Optional[str] = None) -> List[MemoryRecord]:
        records = list(self._records.values())
        if memory_type:
            records = [rec for rec in records if rec.memory_type == memory_type]
        return records

    def stats(self) -> dict:
        counts: Dict[str, int] = {}
        for rec in self._records.values():
            counts[rec.memory_type] = counts.get(rec.memory_type, 0) + 1
        return {"num_records": len(self._records), "counts": counts}

    def _add(
        self,
        memory_type: str,
        content: str,
        tags: Sequence[str],
        metadata: Dict[str, Any],
        strength: float = 1.0,
    ) -> MemoryRecord:
        self._counter += 1
        memory_id = f"mem_{self._counter:08d}"
        vec = self._encode(content)
        embedding = vec.astype(float).tolist() if (self.store_embeddings and vec is not None) else None
        rec = MemoryRecord(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            tags=list(dict.fromkeys(tags)),
            metadata=metadata,
            strength=float(strength),
            embedding=embedding,
        )
        self._records[memory_id] = rec
        return rec

    def _record_vector(self, rec: MemoryRecord) -> Optional[np.ndarray]:
        if rec.embedding is not None:
            return np.array(rec.embedding, dtype=np.float32)
        return self._encode(rec.content)

    def _encode(self, text: str) -> Optional[np.ndarray]:
        if self.text_encoder is None:
            return None
        return self.text_encoder.encode(text)

    def _terms(self, text: str) -> List[str]:
        if self.text_encoder is not None:
            return self.text_encoder.tokenize(text)
        return [t.lower() for t in text.replace("?", " ").replace(".", " ").split()]

    def _keyword_score(self, q_terms: set, rec: MemoryRecord) -> float:
        if not q_terms:
            return 0.0
        r_terms = set(self._terms(rec.content)).union(rec.tags)
        return len(q_terms.intersection(r_terms)) / max(1, len(q_terms))

    def _compact(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.astype(float).round(5).tolist()
        if isinstance(value, (list, tuple)):
            return [self._compact(v) for v in value[:32]]
        if isinstance(value, dict):
            return {str(k): self._compact(v) for k, v in list(value.items())[:32]}
        return value
