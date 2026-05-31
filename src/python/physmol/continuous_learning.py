"""Continuous Learning Module for PHYSMOL.

Enables the system to learn from every interaction, improving over time.

Learning mechanisms:
  1. Vocabulary expansion: learn new words from user input
  2. Pattern learning: learn grammar patterns from successful responses
  3. Concept learning: learn new concepts from dialogue context
  4. Feedback reinforcement: strengthen good patterns, weaken bad ones
  5. Style adaptation: adapt to user's language style over time

Key difference from LLMs:
  - LLMs are frozen after training
  - This system learns continuously from every conversation
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


@dataclass
class InteractionRecord:
    """Record of a single interaction."""
    timestamp: float
    user_input: str
    system_response: str
    intent: str
    concepts: List[str]
    feedback: str = ""  # "good", "bad", ""
    corrected_response: str = ""  # user's correction if feedback is "bad"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningStats:
    """Statistics about learning progress."""
    total_interactions: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    new_words_learned: int = 0
    new_concepts_learned: int = 0
    new_patterns_learned: int = 0
    vocabulary_size: int = 0
    pattern_count: int = 0


class ContinuousLearner:
    """Continuous learning from user interactions.

    This module sits between the user interface and the language modules,
    capturing every interaction and learning from it.

    Usage:
        learner = ContinuousLearner(ci)

        # Process each interaction
        response = learner.interact("What is gravity?")

        # Provide feedback
        learner.record_feedback("good")  # or "bad", with optional correction

        # Save learned knowledge
        learner.save("./learning_data")
    """

    def __init__(self, cognitive_interface):
        """Initialize with a CognitiveInterface instance."""
        self.ci = cognitive_interface

        # Interaction history
        self._history: List[InteractionRecord] = []

        # Learning state
        self._stats = LearningStats()

        # Pattern success rates: {pattern: (success, total)}
        self._pattern_success: Dict[str, Tuple[int, int]] = {}

        # User style profile
        self._user_style = {
            "preferred_length": "medium",  # short, medium, long
            "formality": "neutral",  # formal, neutral, casual
            "language": "zh",  # zh, en, mixed
        }

        # Pending feedback (waiting for user feedback on last interaction)
        self._pending_record: Optional[InteractionRecord] = None

    def interact(self, user_input: str) -> str:
        """Process a user interaction and learn from it.

        Args:
            user_input: user's text input

        Returns: system response
        """
        # Step 1: Learn new words from user input
        self._learn_vocabulary(user_input)

        # Step 2: Extract concepts
        concepts = self._extract_concepts(user_input)

        # Step 3: Detect intent
        intent = self._detect_intent(user_input)

        # Step 4: Generate response
        # Always generate via standard pipeline first
        response = self.ci.query(user_input)

        # If Broca is trained, try to use it for better responses
        if self.ci.broca._trained and len(self.ci.broca.grammar._patterns) > 3:
            broca_response = self.ci.broca.produce(concepts, intent)
            # Use Broca's response if it's substantial
            if len(broca_response) >= len(response) * 0.5:
                response = broca_response

        # Step 5: Record the interaction
        record = InteractionRecord(
            timestamp=time.time(),
            user_input=user_input,
            system_response=response,
            intent=intent,
            concepts=concepts,
        )
        self._history.append(record)
        self._pending_record = record

        # Step 6: Update stats
        self._stats.total_interactions += 1

        # Step 7: Learn from this interaction (self-supervised)
        self._learn_from_interaction(record)

        return response

    def record_feedback(self, feedback: str, correction: str = ""):
        """Record user feedback on the last interaction.

        Args:
            feedback: "good" or "bad"
            correction: corrected response if feedback is "bad"
        """
        if self._pending_record is None:
            return

        self._pending_record.feedback = feedback
        self._pending_record.corrected_response = correction

        if feedback == "good":
            self._stats.positive_feedback += 1
            self._reinforce_pattern(self._pending_record)
        elif feedback == "bad":
            self._stats.negative_feedback += 1
            if correction:
                self._learn_from_correction(self._pending_record, correction)

        self._pending_record = None

    def _learn_vocabulary(self, text: str):
        """Learn new words from text."""
        # Tokenize
        tokens = self.ci.text_encoder.tokenize(text.lower())

        new_words = 0
        for token in tokens:
            if not self.ci.text_encoder.lexicon.has_word(token):
                # New word - add to vocabulary
                self.ci.text_encoder.lexicon.get_vector(token)
                new_words += 1

        self._stats.new_words_learned += new_words
        self._stats.vocabulary_size = self.ci.text_encoder.lexicon.vocabulary_size

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract concepts from text."""
        tokens = self.ci.text_encoder.tokenize(text.lower())
        concepts = []

        # Check against known concepts
        for token in tokens:
            # Check VSA primitives
            if self.ci.primitives.get_by_id(token) is not None:
                concepts.append(token)

            # Check learned concepts
            concept_info = self.ci.knowledge.get_concept(token)
            if concept_info:
                concepts.append(token)

        return concepts

    def _detect_intent(self, text: str) -> str:
        """Detect intent from text."""
        parsed = self.ci.semantic_parser.parse_query(text)
        return parsed.get("intent", "unknown")

    def _learn_from_interaction(self, record: InteractionRecord):
        """Learn from a single interaction (self-supervised)."""

        # 1. Learn vocabulary from both input and response
        for word in self.ci.text_encoder.tokenize(record.system_response.lower()):
            self.ci.broca.vocab.add_word(word)

        # 2. Learn concept-word associations
        for concept in record.concepts:
            words = self.ci.text_encoder.tokenize(record.user_input.lower())
            self.ci.broca.grammar.learn_concept_words(concept, words)

            # Also learn from response words
            response_words = self.ci.text_encoder.tokenize(record.system_response.lower())
            self.ci.broca.grammar.learn_concept_words(concept, response_words)

        # 3. Learn grammar patterns from the response
        self.ci.broca.grammar.learn_from_sentence(
            record.system_response, record.intent)

        # 4. Mark Broca as trained if it has enough patterns
        if not self.ci.broca._trained and len(self.ci.broca.grammar._patterns) > 0:
            self.ci.broca._trained = True

        # 5. Track pattern success
        pattern_key = f"{record.intent}:{record.concepts}"
        if pattern_key not in self._pattern_success:
            self._pattern_success[pattern_key] = (0, 0)
        success, total = self._pattern_success[pattern_key]
        self._pattern_success[pattern_key] = (success, total + 1)

    def _reinforce_pattern(self, record: InteractionRecord):
        """Reinforce a successful pattern."""
        # Boost pattern frequency in Broca
        self.ci.broca.grammar.learn_from_sentence(
            record.system_response, "good_response")

        # Update success rate
        pattern_key = f"{record.intent}:{record.concepts}"
        if pattern_key in self._pattern_success:
            success, total = self._pattern_success[pattern_key]
            self._pattern_success[pattern_key] = (success + 1, total)

    def _learn_from_correction(self, record: InteractionRecord,
                                correction: str):
        """Learn from user correction."""
        # Add correction words to vocabulary
        self._learn_vocabulary(correction)

        # Learn the corrected pattern
        self.ci.broca.grammar.learn_from_sentence(correction, record.intent)

        # Store the correction as a better example
        self.ci.broca.grammar.learn_from_sentence(correction, "good_response")

        # Update concept-word mappings with correction
        for concept in record.concepts:
            words = self.ci.text_encoder.tokenize(correction.lower())
            self.ci.broca.grammar.learn_concept_words(concept, words)

    def get_stats(self) -> dict:
        """Get learning statistics."""
        return {
            "total_interactions": self._stats.total_interactions,
            "positive_feedback": self._stats.positive_feedback,
            "negative_feedback": self._stats.negative_feedback,
            "new_words_learned": self._stats.new_words_learned,
            "vocabulary_size": self._stats.vocabulary_size,
            "pattern_count": len(self._pattern_success),
            "success_rate": (
                self._stats.positive_feedback /
                max(1, self._stats.positive_feedback + self._stats.negative_feedback)
            ),
            "user_style": self._user_style,
        }

    def get_history(self, limit: int = 100) -> List[dict]:
        """Get interaction history."""
        records = self._history[-limit:]
        return [
            {
                "timestamp": r.timestamp,
                "user_input": r.user_input,
                "system_response": r.system_response,
                "intent": r.intent,
                "concepts": r.concepts,
                "feedback": r.feedback,
            }
            for r in records
        ]

    def save(self, path: str):
        """Save learning state to disk."""
        os.makedirs(path, exist_ok=True)

        # Save interaction history
        history_path = os.path.join(path, "interactions.jsonl")
        with open(history_path, 'w', encoding='utf-8') as f:
            for record in self._history:
                data = {
                    "timestamp": record.timestamp,
                    "user_input": record.user_input,
                    "system_response": record.system_response,
                    "intent": record.intent,
                    "concepts": record.concepts,
                    "feedback": record.feedback,
                    "corrected_response": record.corrected_response,
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        # Save pattern success rates
        pattern_path = os.path.join(path, "patterns.json")
        with open(pattern_path, 'w', encoding='utf-8') as f:
            json.dump(self._pattern_success, f, indent=2)

        # Save stats
        stats_path = os.path.join(path, "stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.get_stats(), f, indent=2)

        # Save Broca module
        broca_path = os.path.join(path, "broca")
        self.ci.broca.save(broca_path)

    def load(self, path: str):
        """Load learning state from disk."""
        # Load pattern success rates
        pattern_path = os.path.join(path, "patterns.json")
        if os.path.exists(pattern_path):
            with open(pattern_path, 'r', encoding='utf-8') as f:
                self._pattern_success = json.load(f)

        # Load Broca module
        broca_path = os.path.join(path, "broca")
        if os.path.exists(broca_path):
            self.ci.broca.load(broca_path)

        # Load interaction history
        history_path = os.path.join(path, "interactions.jsonl")
        if os.path.exists(history_path):
            self._history = []
            with open(history_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    record = InteractionRecord(**data)
                    self._history.append(record)

            self._stats.total_interactions = len(self._history)

    def interactive_session(self):
        """Start an interactive learning session.

        This is a simple CLI for testing continuous learning.
        """
        print("=" * 60)
        print("PHYSMOL Continuous Learning Session")
        print("=" * 60)
        print("Type 'quit' to exit, 'stats' to see learning stats")
        print("After each response, type 'good' or 'bad' for feedback")
        print()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                break

            if user_input.lower() == 'stats':
                stats = self.get_stats()
                print(f"\nStats: {json.dumps(stats, indent=2)}\n")
                continue

            # Generate response
            response = self.interact(user_input)
            print(f"PHYSMOL: {response}")

            # Get feedback
            try:
                feedback = input("Feedback (good/bad/enter to skip): ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if feedback == 'good':
                self.record_feedback('good')
                print("(Learned: positive feedback)")
            elif feedback == 'bad':
                correction = input("Correct response (enter to skip): ").strip()
                self.record_feedback('bad', correction)
                print("(Learned: correction recorded)")

        print("\nSession ended. Saving learning state...")
        self.save("./learning_data")
        print("Saved.")
