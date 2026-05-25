"""TextToVSA: Natural Language -> VSA Vector Encoder.

Two modes:
  1. Word-level VSA composition (always available, no external deps)
     - Each word gets a random bipolar vector from a persistent lexicon
     - Sentence = bundle of positional word vectors
  2. Sentence-transformer projection (optional, better quality)
     - Uses a lightweight pretrained model (all-MiniLM-L6-v2)
     - Projects 384-dim embeddings into VSA space

The word-level mode is the default for PHYSMOL's "from scratch" philosophy --
the system builds its own word representations through interaction, not by
relying on pretrained models.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import re


CHINESE_TERMS = [
    "公平", "正义", "法律", "惩罚", "处罚", "犯罪者", "犯罪", "民主", "自由",
    "平等", "权利", "奴隶制", "奴役", "错误", "错的", "认为", "相信", "知道",
    "意图", "情绪", "想要", "希望", "打算", "计划", "好奇", "学习", "模型",
    "物体", "球", "方块", "立方体", "红色", "蓝色", "重", "轻", "下落",
    "掉落", "推动", "碰撞", "摩擦", "重力", "弹性", "解释", "如果", "为什么",
    "什么", "你好", "你是谁",
]


class WordLexicon:
    """Persistent word-to-VSA-vector lexicon.

    New words are assigned random bipolar vectors on first encounter.
    Over time, vectors can be refined through contrastive learning
    (alignment with physical experience).
    """

    def __init__(self, vsa_dim: int = 10000, seed: int = 42):
        self.vsa_dim = vsa_dim
        self.rng = np.random.RandomState(seed)
        self._vectors: Dict[str, np.ndarray] = {}

        # Positional encoding vectors (for word order)
        self._pos_vectors: List[np.ndarray] = [
            self._rand_bipolar() for _ in range(64)
        ]

    def _rand_bipolar(self) -> np.ndarray:
        return (self.rng.randint(0, 2, self.vsa_dim).astype(np.float32) * 2 - 1)

    def get_vector(self, word: str) -> np.ndarray:
        """Get or create a VSA vector for a word."""
        if word not in self._vectors:
            self._vectors[word] = self._rand_bipolar()
        return self._vectors[word]

    def has_word(self, word: str) -> bool:
        return word in self._vectors

    def set_vector(self, word: str, vec: np.ndarray):
        """Set a word vector explicitly (e.g., from alignment training)."""
        self._vectors[word] = vec

    def get_position_vector(self, pos: int) -> np.ndarray:
        """Get positional encoding vector for word position."""
        idx = min(pos, len(self._pos_vectors) - 1)
        return self._pos_vectors[idx]

    @property
    def vocabulary_size(self) -> int:
        return len(self._vectors)

    def words(self) -> List[str]:
        return list(self._vectors.keys())


class TextToVSA:
    """Encode natural language text into VSA vectors.

    Sentence encoding:
      sentence_vec = Σ (word_vec_i ⊗ position_vec_i)

    This preserves word order through positional binding.
    """

    def __init__(self, vsa_dim: int = 10000, seed: int = 42):
        self.vsa_dim = vsa_dim
        self.lexicon = WordLexicon(vsa_dim, seed)

        # Pre-populate common physics words
        self._init_physics_vocabulary()

        # Optional: sentence-transformer encoder
        self._st_model = None
        self._st_proj = None

    def _init_physics_vocabulary(self):
        """Pre-populate words relevant to physical reasoning."""
        physics_words = [
            # Objects
            "ball", "cube", "block", "cylinder", "sphere", "ramp", "slope",
            "table", "floor", "wall", "spring", "pendulum", "rope", "chain",
            # Properties
            "red", "blue", "green", "yellow", "white", "black", "orange",
            "heavy", "light", "big", "small", "hard", "soft", "smooth", "rough",
            "round", "square", "flat", "tall", "short", "wide", "narrow",
            "elastic", "rigid", "stiff", "flexible", "brittle",
            # Materials
            "metal", "wood", "plastic", "rubber", "glass", "stone", "fabric",
            # Physics concepts
            "fall", "drop", "roll", "bounce", "slide", "push", "pull", "lift",
            "collide", "crash", "hit", "break", "bend", "stretch", "compress",
            "force", "energy", "momentum", "velocity", "speed", "acceleration",
            "gravity", "friction", "mass", "weight", "inertia",
            # Spatial
            "up", "down", "left", "right", "top", "bottom", "above", "below",
            "behind", "front", "inside", "outside", "near", "far", "on", "under",
            # Actions
            "what", "where", "when", "how", "why", "if", "then", "would",
            "will", "can", "could", "should", "might", "does", "did",
            "put", "place", "move", "stop", "start", "go", "come",
            # Quantifiers
            "more", "less", "most", "least", "very", "extremely", "slightly",
            "all", "some", "none", "each", "every", "any",
            # Connectors
            "and", "or", "but", "not", "with", "without", "from", "to",
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "it", "its", "this", "that", "these", "those",
        ]
        physics_words.extend(CHINESE_TERMS)
        for word in physics_words:
            self.lexicon.get_vector(word)

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        text = text.lower().strip()
        raw_tokens = re.findall(r'[a-z]+|[0-9]+|[\u4e00-\u9fff]+', text)
        tokens: List[str] = []
        for token in raw_tokens:
            if re.fullmatch(r'[\u4e00-\u9fff]+', token):
                tokens.extend(self._tokenize_chinese(token))
            else:
                tokens.append(token)
        return tokens

    def _tokenize_chinese(self, text: str) -> List[str]:
        """Greedy phrase tokenizer for the small built-in Chinese vocabulary."""
        tokens: List[str] = []
        terms = sorted(CHINESE_TERMS, key=len, reverse=True)
        i = 0
        while i < len(text):
            matched = None
            for term in terms:
                if text.startswith(term, i):
                    matched = term
                    break
            if matched:
                tokens.append(matched)
                i += len(matched)
            else:
                tokens.append(text[i])
                i += 1
        return tokens

    def encode_tokens(self, tokens: List[str]) -> np.ndarray:
        """Encode a list of tokens into a single VSA vector.

        Uses positional binding: sentence = Σ (word_i ⊗ pos_i)
        """
        if not tokens:
            return np.zeros(self.vsa_dim, dtype=np.float32)

        vec = np.zeros(self.vsa_dim, dtype=np.float32)
        for i, token in enumerate(tokens):
            word_vec = self.lexicon.get_vector(token)
            pos_vec = self.lexicon.get_position_vector(i)
            vec += word_vec * pos_vec  # bind word with position

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def encode(self, text: str) -> np.ndarray:
        """Encode a text string into a VSA vector."""
        tokens = self.tokenize(text)
        return self.encode_tokens(tokens)

    def encode_word(self, word: str) -> np.ndarray:
        """Get the VSA vector for a single word."""
        return self.lexicon.get_vector(word.lower())

    def try_init_sentence_transformer(self, model_name: str = 'all-MiniLM-L6-v2'
                                       ) -> bool:
        """Try to initialize a sentence-transformer encoder.

        Returns True if successful, False otherwise.
        """
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            st_dim = model.get_sentence_embedding_dimension()

            # Create random projection from ST space to VSA space
            rng = np.random.RandomState(42)
            proj = rng.randn(st_dim, self.vsa_dim).astype(np.float32)
            proj /= np.linalg.norm(proj, axis=1, keepdims=True)

            self._st_model = model
            self._st_proj = proj
            return True
        except (ImportError, Exception):
            return False

    def encode_with_transformer(self, text: str) -> np.ndarray:
        """Encode using sentence-transformer + projection.

        Falls back to word-level if transformer not available.
        """
        if self._st_model is None:
            return self.encode(text)

        embedding = self._st_model.encode(text, convert_to_numpy=True)
        vec = embedding @ self._st_proj
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32)
