"""Progress logging utilities for long-running PHYSMOL training."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import time
from typing import Any, Dict, List, Optional


@dataclass
class ProgressEvent:
    step: int
    total_steps: int
    phase: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        pct = 0.0 if self.total_steps <= 0 else min(100.0, 100.0 * self.step / self.total_steps)
        return {
            "step": self.step,
            "total_steps": self.total_steps,
            "percent": pct,
            "phase": self.phase,
            "metrics": self.metrics,
            "message": self.message,
            "timestamp": self.timestamp,
        }


class ProgressLogger:
    """Write progress snapshots and append-only event logs."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.snapshot_path = os.path.join(out_dir, "progress.json")
        self.events_path = os.path.join(out_dir, "progress.jsonl")

    def log(
        self,
        step: int,
        total_steps: int,
        phase: str,
        metrics: Optional[Dict[str, Any]] = None,
        message: str = "",
    ) -> dict:
        event = ProgressEvent(
            step=step,
            total_steps=total_steps,
            phase=phase,
            metrics=metrics or {},
            message=message,
        ).to_dict()
        with open(self.snapshot_path, "w", encoding="utf-8") as f:
            json.dump(event, f, ensure_ascii=False, indent=2)
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def read_latest(self) -> dict:
        if not os.path.exists(self.snapshot_path):
            return {
                "step": 0,
                "total_steps": 0,
                "percent": 0.0,
                "phase": "idle",
                "metrics": {},
                "message": "No training progress has been written yet.",
                "timestamp": time.time(),
            }
        with open(self.snapshot_path, "r", encoding="utf-8") as f:
            return json.load(f)
