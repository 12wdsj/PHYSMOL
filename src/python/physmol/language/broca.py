"""Broca's Area - Language Production Module.

Simulates the brain's Broca's area: converts conceptual representations
(VSA vectors) into sequential language output.

Architecture:
  Concept Vector → Vocabulary Access → Grammar Encoding → Word Sequence

This is NOT an LLM. It's a lightweight recurrent network that learns
to "articulate" VSA concepts through experience.

Training:
  - Phase 1: Vocabulary learning (concept → word mapping)
  - Phase 2: Grammar learning (word ordering from dialogue data)
  - Phase 3: Interactive refinement (learn from user feedback)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


# ---------------------------------------------------------------------------
# Vocabulary (Wernicke's Area - Word Access)
# ---------------------------------------------------------------------------

class Vocabulary:
    """Word vocabulary with VSA concept mapping.

    Maps between word IDs and words, and between VSA concepts and words.
    """

    def __init__(self):
        # Word ↔ ID mapping
        self._word2id: Dict[str, int] = {}
        self._id2word: Dict[int, str] = {}

        # Special tokens
        self.pad_id = self._add_token("<PAD>")
        self.start_id = self._add_token("<START>")
        self.end_id = self._add_token("<END>")
        self.unk_id = self._add_token("<UNK>")

    def _add_token(self, token: str) -> int:
        tid = len(self._word2id)
        self._word2id[token] = tid
        self._id2word[tid] = token
        return tid

    def add_word(self, word: str) -> int:
        """Add a word to vocabulary, return its ID."""
        if word in self._word2id:
            return self._word2id[word]
        return self._add_token(word)

    def get_id(self, word: str) -> int:
        """Get word ID, returns UNK if not found."""
        return self._word2id.get(word, self.unk_id)

    def get_word(self, tid: int) -> str:
        """Get word from ID."""
        return self._id2word.get(tid, "<UNK>")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = self._tokenize(text)
        return [self.start_id] + [self.get_id(t) for t in tokens] + [self.end_id]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        words = []
        for tid in ids:
            if tid == self.end_id:
                break
            if tid not in (self.pad_id, self.start_id):
                word = self.get_word(tid)
                if word != "<UNK>":
                    words.append(word)
        # For Chinese: join without spaces
        # For English: join with spaces
        if any('\u4e00' <= c <= '\u9fff' for w in words for c in w):
            return "".join(words)
        return " ".join(words)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer."""
        text = text.lower().strip()
        # Split by whitespace and punctuation, keep Chinese characters separate
        tokens = []
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                tokens.append(char)
            elif char.isalnum():
                tokens.append(char)
            elif char in '.,!?;:':
                tokens.append(char)
        # Merge consecutive ASCII characters
        merged = []
        current = ""
        for t in tokens:
            if '\u4e00' <= t <= '\u9fff':
                if current:
                    merged.append(current)
                    current = ""
                merged.append(t)
            elif t.isalnum():
                current += t
            else:
                if current:
                    merged.append(current)
                    current = ""
                merged.append(t)
        if current:
            merged.append(current)
        return merged

    @property
    def size(self) -> int:
        return len(self._word2id)

    def save(self, path: str):
        """Save vocabulary to file."""
        data = {"word2id": self._word2id}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """Load vocabulary from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self._word2id = data["word2id"]
        self._id2word = {int(v): k for k, v in self._word2id.items()}


# ---------------------------------------------------------------------------
# Grammar Encoder (Broca's Area - Sequence Generation)
# ---------------------------------------------------------------------------

class GrammarEncoder:
    """Simple sequence generator using learned patterns.

    Instead of a neural network, this uses a pattern-based approach:
    1. Learn sentence templates from data
    2. Fill templates with concepts
    3. Apply grammar rules for fluency
    """

    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab

        # Learned patterns: {intent: [(template, frequency)]}
        self._patterns: Dict[str, List[Tuple[str, int]]] = {}

        # Word transition probabilities: {prev_word: {next_word: prob}}
        self._transitions: Dict[str, Dict[str, float]] = {}

        # Concept → word mappings
        self._concept_words: Dict[str, List[str]] = {}

    def learn_from_sentence(self, sentence: str, intent: str = "general"):
        """Learn patterns from a sentence."""
        tokens = self.vocab._tokenize(sentence.lower())

        # Store as template pattern
        template = " ".join(tokens)
        if intent not in self._patterns:
            self._patterns[intent] = []

        # Check if similar pattern exists
        found = False
        for i, (t, freq) in enumerate(self._patterns[intent]):
            if self._similarity(t, template) > 0.7:
                self._patterns[intent][i] = (t, freq + 1)
                found = True
                break

        if not found:
            self._patterns[intent].append((template, 1))

        # Update transition probabilities
        for i in range(len(tokens) - 1):
            if tokens[i] not in self._transitions:
                self._transitions[tokens[i]] = {}
            next_word = tokens[i + 1]
            self._transitions[tokens[i]][next_word] = \
                self._transitions[tokens[i]].get(next_word, 0) + 1

    def learn_concept_words(self, concept: str, words: List[str]):
        """Learn which words are associated with a concept."""
        self._concept_words[concept] = words

    def generate(self, concepts: List[str], intent: str = "general",
                 max_len: int = 30) -> str:
        """Generate a sentence from concepts.

        Strategy:
          1. Find best matching template for the intent
          2. Fill template slots with concept-related words
          3. Use transition probabilities for fluency
        """
        # Get candidate words from concepts
        candidate_words = []
        for concept in concepts:
            if concept in self._concept_words:
                candidate_words.extend(self._concept_words[concept])
            # Also try to find words from vocabulary
            for word in self.vocab._word2id:
                if concept.lower() in word or word in concept.lower():
                    candidate_words.append(word)

        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for w in candidate_words:
            if w not in seen:
                seen.add(w)
                unique_words.append(w)
        candidate_words = unique_words

        # Find best template
        template = self._find_template(intent, candidate_words)

        if template:
            # Fill template with concept words
            return self._fill_template(template, candidate_words)
        else:
            # Fallback: generate from transitions
            return self._generate_from_transitions(concepts, max_len)

    def _find_template(self, intent: str, words: List[str]) -> Optional[str]:
        """Find the best matching template for given intent and words."""
        # Try the specified intent first, then fall back to general
        for try_intent in [intent, "general"]:
            if try_intent not in self._patterns or not self._patterns[try_intent]:
                continue

            best_template = None
            best_score = -1

            for template, freq in self._patterns[try_intent]:
                # Score based on word overlap and frequency
                template_lower = template.lower()
                score = 0

                # Check if any concept word appears in the template
                for word in words:
                    if word.lower() in template_lower:
                        score += 2  # Direct match gets high score
                    elif any(c in template_lower for c in word.lower()):
                        score += 1  # Partial match

                # Add frequency bonus
                score += np.log1p(freq) * 0.1

                if score > best_score:
                    best_score = score
                    best_template = template

            if best_template and best_score > 0:
                return best_template

        return None

    def _fill_template(self, template: str, words: List[str]) -> str:
        """Fill a template with available words."""
        # Try to find the best matching template from learned patterns
        for word in words:
            # Find a template that contains similar concepts
            for intent, patterns in self._patterns.items():
                for pattern, freq in patterns:
                    if word.lower() in pattern.lower():
                        # Use this pattern
                        return pattern

        # Fallback: join the most relevant words
        if words:
            # Sort by relevance (longer words are usually more specific)
            sorted_words = sorted(words, key=len, reverse=True)
            # For Chinese: join without spaces
            # For English: join with spaces
            if any('\u4e00' <= c <= '\u9fff' for w in sorted_words for c in w):
                return "".join(sorted_words[:5])
            return " ".join(sorted_words[:5])

        return template

    def _generate_from_transitions(self, concepts: List[str],
                                    max_len: int) -> str:
        """Generate text using word transition probabilities."""
        # Start with a concept word
        if not concepts:
            return ""

        start_word = concepts[0].lower()
        if start_word not in self._transitions:
            # Try to find a similar word
            for word in self._transitions:
                if any(c in word for c in concepts):
                    start_word = word
                    break

        if start_word not in self._transitions:
            # For Chinese: join without spaces
            if any('\u4e00' <= c <= '\u9fff' for w in concepts for c in w):
                return "".join(concepts)
            return " ".join(concepts)

        # Generate word by word
        result = [start_word]
        current = start_word

        for _ in range(max_len):
            if current not in self._transitions:
                break

            # Choose next word based on probability
            next_words = self._transitions[current]
            if not next_words:
                break

            # Weighted random selection
            words = list(next_words.keys())
            probs = np.array(list(next_words.values()), dtype=np.float32)
            probs /= probs.sum()

            next_word = np.random.choice(words, p=probs)
            result.append(next_word)
            current = next_word

        # For Chinese: join without spaces
        if any('\u4e00' <= c <= '\u9fff' for w in result for c in w):
            return "".join(result)
        return " ".join(result)

    def _similarity(self, s1: str, s2: str) -> float:
        """Simple string similarity."""
        if s1 == s2:
            return 1.0
        # Character overlap
        set1 = set(s1)
        set2 = set(s2)
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / max(len(set1), len(set2))

    def get_stats(self) -> dict:
        """Get grammar encoder statistics."""
        total_patterns = sum(len(v) for v in self._patterns.values())
        return {
            "intents": len(self._patterns),
            "total_patterns": total_patterns,
            "transition_words": len(self._transitions),
            "concept_mappings": len(self._concept_words),
        }


# ---------------------------------------------------------------------------
# Broca Module (Complete Language Production System)
# ---------------------------------------------------------------------------

class BrocaModule:
    """Complete language production system, simulating Broca's area.

    Combines vocabulary access and grammar encoding to convert
    VSA concepts into natural language.

    Usage:
        broca = BrocaModule(vsa_dim=4096)

        # Learn from dialogue
        broca.train_from_dialogue(concepts, sentences)

        # Generate language
        sentence = broca.produce(concept_vec, intent="explanation")
    """

    def __init__(self, vsa_dim: int = 4096):
        self.vsa_dim = vsa_dim

        # Components
        self.vocab = Vocabulary()
        self.grammar = GrammarEncoder(self.vocab)

        # Concept vector cache
        self._concept_cache: Dict[str, np.ndarray] = {}

        # Training state
        self._trained = False
        self._training_samples = 0

    def train_from_dialogue(self, dialogues: List[Dict[str, str]]):
        """Train from dialogue data.

        Each dialogue is a dict with:
          - "input": user input
          - "output": expected response
          - "intent": optional intent label
          - "concepts": optional list of concept strings
        """
        for dialogue in dialogues:
            response = dialogue.get("output", "")
            intent = dialogue.get("intent", "general")
            concepts = dialogue.get("concepts", [])

            # Add words to vocabulary
            for word in self.vocab._tokenize(response.lower()):
                self.vocab.add_word(word)

            # Learn grammar patterns
            self.grammar.learn_from_sentence(response, intent)

            # Learn concept-word mappings
            if concepts:
                words = self.vocab._tokenize(response.lower())
                for concept in concepts:
                    self.grammar.learn_concept_words(concept, words)

            self._training_samples += 1

        self._trained = True

    def train_from_file(self, path: str, format: str = "auto"):
        """Train from a data file."""
        if format == "auto":
            if path.endswith(".jsonl"):
                format = "jsonl"
            elif path.endswith(".json"):
                format = "json"
            elif path.endswith(".txt"):
                format = "txt"
            else:
                format = "jsonl"

        if format == "jsonl":
            self._train_from_jsonl(path)
        elif format == "json":
            self._train_from_json(path)
        elif format == "txt":
            self._train_from_txt(path)

    def _train_from_jsonl(self, path: str):
        """Train from JSONL file."""
        dialogues = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Normalize format
                    dialogue = {
                        "input": data.get("instruction", data.get("input", data.get("question", ""))),
                        "output": data.get("output", data.get("response", data.get("answer", ""))),
                        "intent": data.get("intent", "general"),
                    }
                    if dialogue["output"]:
                        dialogues.append(dialogue)
                except json.JSONDecodeError:
                    continue

        self.train_from_dialogue(dialogues)

    def _train_from_json(self, path: str):
        """Train from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            self.train_from_dialogue(data)

    def _train_from_txt(self, path: str):
        """Train from plain text file (one sentence per line)."""
        dialogues = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and len(line) > 5:
                    dialogues.append({
                        "input": "",
                        "output": line,
                        "intent": "general",
                    })
        self.train_from_dialogue(dialogues)

    def produce(self, concepts: List[str], intent: str = "general") -> str:
        """Generate language from concepts.

        Args:
            concepts: list of concept strings
            intent: intent label (question, explanation, etc.)

        Returns: generated sentence
        """
        if not self._trained:
            # Fallback: just join concepts
            return " ".join(concepts)

        return self.grammar.generate(concepts, intent)

    def produce_from_vsa(self, concept_vec: np.ndarray,
                          intent: str = "general") -> str:
        """Generate language from a VSA vector.

        Uses vector similarity to find related concepts,
        then generates language from those concepts.
        """
        # Find related concepts from cache
        related = self._find_related_concepts(concept_vec)

        # Generate language
        return self.produce(related, intent)

    def _find_related_concepts(self, vec: np.ndarray,
                                top_k: int = 5) -> List[str]:
        """Find concepts related to a VSA vector."""
        if not self._concept_cache:
            return []

        similarities = []
        for concept, c_vec in self._concept_cache.items():
            sim = np.dot(vec, c_vec) / (np.linalg.norm(vec) * np.linalg.norm(c_vec) + 1e-8)
            similarities.append((concept, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in similarities[:top_k]]

    def register_concept(self, name: str, vec: np.ndarray):
        """Register a concept vector for later retrieval."""
        self._concept_cache[name] = vec

    def learn_from_interaction(self, user_input: str, response: str,
                                feedback: str = "good"):
        """Learn from a single interaction.

        This enables continuous learning from user feedback.
        """
        # Add response to vocabulary
        for word in self.vocab._tokenize(response.lower()):
            self.vocab.add_word(word)

        # Learn grammar pattern
        self.grammar.learn_from_sentence(response, "conversation")

        # If feedback is positive, reinforce the pattern
        if feedback == "good":
            self.grammar.learn_from_sentence(response, "good_response")
        elif feedback == "bad" and user_input:
            # Store as negative example (avoid this pattern)
            pass

    def get_stats(self) -> dict:
        """Get module statistics."""
        return {
            "vocabulary_size": self.vocab.size,
            "trained": self._trained,
            "training_samples": self._training_samples,
            "concept_cache_size": len(self._concept_cache),
            "grammar": self.grammar.get_stats(),
        }

    def save(self, path: str):
        """Save the Broca module to disk."""
        os.makedirs(path, exist_ok=True)

        # Save vocabulary
        self.vocab.save(os.path.join(path, "vocab.json"))

        # Save grammar patterns
        grammar_data = {
            "patterns": self.grammar._patterns,
            "transitions": {k: dict(v) for k, v in self.grammar._transitions.items()},
            "concept_words": self.grammar._concept_words,
        }
        with open(os.path.join(path, "grammar.json"), 'w', encoding='utf-8') as f:
            json.dump(grammar_data, f, ensure_ascii=False, indent=2)

        # Save stats
        stats = self.get_stats()
        with open(os.path.join(path, "stats.json"), 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

    def load(self, path: str):
        """Load the Broca module from disk."""
        # Load vocabulary
        self.vocab.load(os.path.join(path, "vocab.json"))

        # Load grammar patterns
        with open(os.path.join(path, "grammar.json"), 'r', encoding='utf-8') as f:
            grammar_data = json.load(f)

        self.grammar._patterns = grammar_data.get("patterns", {})
        self.grammar._transitions = grammar_data.get("transitions", {})
        self.grammar._concept_words = grammar_data.get("concept_words", {})

        self._trained = True
