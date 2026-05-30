"""Automatic Knowledge Acquisition Module.

Enables PHYSMOL to learn new concepts from text and interaction,
without requiring MuJoCo simulation.

Core idea: Extract concepts from natural language, create VSA primitives,
and build relationships between concepts through dialogue.

Learning modes:
  1. Text extraction: Parse text to find new concepts
  2. Interactive learning: User teaches concepts directly
  3. Inference: Derive new concepts from existing ones
  4. Cross-domain transfer: Map physical concepts to abstract domains
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .vsa_store import AttributePrimitivePool, RecipeStore


@dataclass
class ConceptCandidate:
    """A potential new concept extracted from text."""
    term: str
    category: str = ""
    context: str = ""
    confidence: float = 0.5
    related_terms: List[str] = field(default_factory=list)


@dataclass
class LearnedConcept:
    """A concept that has been learned and added to VSA."""
    term: str
    category: str
    attr_id: str
    definition: str = ""
    examples: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    source: str = ""  # where this concept was learned from


class KnowledgeAcquisition:
    """Automatic knowledge acquisition from text and interaction.

    This module enables PHYSMOL to grow its concept vocabulary
    without requiring physical simulation.
    """

    def __init__(self, primitives: AttributePrimitivePool,
                 recipe_store: RecipeStore):
        self.primitives = primitives
        self.recipe_store = recipe_store

        # Learned concepts: {term: LearnedConcept}
        self._learned: Dict[str, LearnedConcept] = {}

        # Concept relationships: {term: {related_term: strength}}
        self._relationships: Dict[str, Dict[str, float]] = {}

        # Category inference patterns
        self._category_patterns = self._init_category_patterns()

        # Definition patterns for extraction
        self._definition_patterns = self._init_definition_patterns()

    def _init_category_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for category inference."""
        return {
            "algorithm": [
                r"algorithm", r"sort", r"search", r"traversal",
                r"divide and conquer", r"dynamic programming", r"greedy",
                r"算法", r"排序", r"搜索", r"遍历",
            ],
            "data_structure": [
                r"data structure", r"array", r"list", r"stack", r"queue",
                r"tree", r"graph", r"hash", r"heap",
                r"数据结构", r"数组", r"链表", r"栈", r"队列", r"树", r"图",
            ],
            "concept": [
                r"concept", r"principle", r"theory", r"law",
                r"概念", r"原理", r"理论", r"定律",
            ],
            "object": [
                r"object", r"thing", r"item", r"entity",
                r"物体", r"对象", r"实体",
            ],
            "action": [
                r"action", r"operation", r"process", r"method",
                r"动作", r"操作", r"过程", r"方法",
            ],
            "property": [
                r"property", r"attribute", r"feature", r"characteristic",
                r"属性", r"特征", r"特性",
            ],
        }

    def _init_definition_patterns(self) -> List[Tuple[str, str]]:
        """Initialize patterns for extracting definitions from text.

        Returns: list of (pattern, group_index) tuples
        """
        return [
            # English patterns
            (r"(\w+)\s+is\s+(?:a|an|the)\s+(.+?)(?:\.|,|;|$)", 2),
            (r"(\w+)\s+refers to\s+(.+?)(?:\.|,|;|$)", 2),
            (r"(\w+)\s+means\s+(.+?)(?:\.|,|;|$)", 2),
            (r"(\w+)\s*:\s*(.+?)(?:\.|,|;|$)", 2),
            # Chinese patterns
            (r"(\w+)是(.+?)(?:。|，|；|$)", 2),
            (r"(\w+)指的是(.+?)(?:。|，|；|$)", 2),
            (r"(\w+)：(.+?)(?:。|，|；|$)", 2),
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_concepts(self, text: str) -> List[ConceptCandidate]:
        """Extract potential new concepts from text.

        Uses heuristic patterns to identify terms that might be
        new concepts worth learning.
        """
        candidates = []
        text_lower = text.lower()

        # Extract noun phrases (simple heuristic)
        # Pattern: "the X", "a X", "an X", or capitalized words
        noun_patterns = [
            r"(?:the|a|an)\s+(\w+(?:\s+\w+)?)",  # "the quicksort algorithm"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",   # "Quicksort Algorithm"
            r"(\w+(?:_\w+)+)",                       # "binary_search"
        ]

        found_terms = set()
        for pattern in noun_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                term = match.strip().lower()
                if len(term) > 2 and term not in found_terms:
                    found_terms.add(term)
                    category = self._infer_category(term, text_lower)
                    candidates.append(ConceptCandidate(
                        term=term,
                        category=category,
                        context=text[:200],
                        confidence=0.3,
                    ))

        # Extract terms near definition patterns
        for pattern, def_group in self._definition_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group(1).strip().lower()
                definition = match.group(def_group).strip()
                if term not in found_terms and len(term) > 2:
                    found_terms.add(term)
                    category = self._infer_category(term, text_lower)
                    candidates.append(ConceptCandidate(
                        term=term,
                        category=category,
                        context=definition,
                        confidence=0.7,
                    ))

        return candidates

    def learn_from_text(self, text: str) -> List[LearnedConcept]:
        """Learn new concepts from text input.

        Extracts concepts, creates VSA primitives, and records definitions.
        """
        candidates = self.extract_concepts(text)
        learned = []

        for candidate in candidates:
            if candidate.term in self._learned:
                continue  # Already known

            # Determine category
            category = candidate.category or "concept"

            # Create VSA primitive
            attr_id = f"{category}_{candidate.term.replace(' ', '_')}"
            if not self.primitives.get(category, candidate.term):
                # Add new primitive to the category
                if category not in self.primitives.list_categories():
                    self.primitives.add_category(category, {candidate.term: None})
                else:
                    # Add to existing category
                    self.primitives.add_category(category, {candidate.term: None})

            # Record learned concept
            concept = LearnedConcept(
                term=candidate.term,
                category=category,
                attr_id=attr_id,
                definition=candidate.context,
                source="text_extraction",
            )
            self._learned[candidate.term] = concept
            learned.append(concept)

        return learned

    def learn_concept(self, term: str, category: str = "",
                      definition: str = "", examples: Optional[List[str]] = None,
                      related: Optional[List[str]] = None) -> LearnedConcept:
        """Explicitly learn a new concept (user teaching).

        Args:
            term: the concept term
            category: VSA category (auto-inferred if empty)
            definition: what this concept means
            examples: example usages
            related: related concepts
        """
        term_lower = term.lower().strip()

        # Auto-infer category if not provided
        if not category:
            category = self._infer_category(term_lower, definition)

        # Create VSA primitive
        attr_id = f"{category}_{term_lower.replace(' ', '_')}"
        if category not in self.primitives.list_categories():
            self.primitives.add_category(category, {term_lower: None})
        else:
            # Check if already exists
            existing = self.primitives.get(category, term_lower)
            if existing is None:
                self.primitives.add_category(category, {term_lower: None})

        # Record learned concept
        concept = LearnedConcept(
            term=term_lower,
            category=category,
            attr_id=attr_id,
            definition=definition,
            examples=examples or [],
            related_concepts=related or [],
            source="user_teaching",
        )
        self._learned[term_lower] = concept

        # Build relationships
        if related:
            for rel_term in related:
                self._add_relationship(term_lower, rel_term.lower(), 0.8)

        return concept

    def get_concept(self, term: str) -> Optional[LearnedConcept]:
        """Get a learned concept by term."""
        return self._learned.get(term.lower())

    def list_concepts(self, category: Optional[str] = None) -> List[LearnedConcept]:
        """List all learned concepts, optionally filtered by category."""
        if category:
            return [c for c in self._learned.values() if c.category == category]
        return list(self._learned.values())

    def find_related(self, term: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find concepts related to the given term."""
        term_lower = term.lower()
        if term_lower not in self._relationships:
            return []

        related = self._relationships[term_lower]
        sorted_related = sorted(related.items(), key=lambda x: x[1], reverse=True)
        return sorted_related[:top_k]

    def infer_concept(self, term: str, context: str = "") -> Optional[LearnedConcept]:
        """Infer a new concept from existing knowledge.

        Uses relationships and category patterns to derive new concepts.
        """
        term_lower = term.lower()

        # Check if already known
        if term_lower in self._learned:
            return self._learned[term_lower]

        # Try to infer from context
        category = self._infer_category(term_lower, context)

        # Check if similar concepts exist
        similar = self._find_similar_terms(term_lower)
        if similar:
            # Use the most similar known concept as a base
            base_term, similarity = similar[0]
            base_concept = self._learned.get(base_term)
            if base_concept:
                # Create new concept based on the similar one
                return self.learn_concept(
                    term=term_lower,
                    category=base_concept.category,
                    definition=f"Similar to {base_term}: {base_concept.definition}",
                    related=[base_term],
                )

        # Create a new concept with inferred category
        return self.learn_concept(
            term=term_lower,
            category=category,
            definition=context or f"A {category} concept: {term}",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_category(self, term: str, context: str) -> str:
        """Infer the category of a term from context."""
        combined = f"{term} {context}".lower()

        # Score each category
        scores: Dict[str, float] = {}
        for category, patterns in self._category_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in combined:
                    score += 1
            if score > 0:
                scores[category] = score

        if scores:
            return max(scores, key=scores.get)

        return "concept"  # Default category

    def _add_relationship(self, term1: str, term2: str, strength: float):
        """Add a bidirectional relationship between two concepts."""
        if term1 not in self._relationships:
            self._relationships[term1] = {}
        if term2 not in self._relationships:
            self._relationships[term2] = {}

        self._relationships[term1][term2] = strength
        self._relationships[term2][term1] = strength

    def _find_similar_terms(self, term: str) -> List[Tuple[str, float]]:
        """Find terms similar to the given term using VSA resonance."""
        # Encode the term
        from .language.text_encoder import TextToVSA
        encoder = TextToVSA(self.primitives.vsa_dim)
        term_vec = encoder.encode(term)

        # Find similar in learned concepts
        similarities = []
        for known_term, concept in self._learned.items():
            known_vec = self.primitives.get_by_id(concept.attr_id)
            if known_vec is not None:
                # Compute cosine similarity
                dot = np.dot(term_vec, known_vec)
                norm = np.linalg.norm(term_vec) * np.linalg.norm(known_vec)
                if norm > 0:
                    sim = float(dot / norm)
                    similarities.append((known_term, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:5]

    def get_stats(self) -> dict:
        """Get statistics about learned knowledge."""
        categories = {}
        for concept in self._learned.values():
            categories[concept.category] = categories.get(concept.category, 0) + 1

        return {
            "total_concepts": len(self._learned),
            "categories": categories,
            "relationships": sum(len(r) for r in self._relationships.values()) // 2,
            "sources": {
                "text_extraction": sum(1 for c in self._learned.values()
                                       if c.source == "text_extraction"),
                "user_teaching": sum(1 for c in self._learned.values()
                                     if c.source == "user_teaching"),
                "inference": sum(1 for c in self._learned.values()
                                 if c.source == "inference"),
            },
        }
