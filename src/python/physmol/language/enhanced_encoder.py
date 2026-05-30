"""Enhanced Language Encoder for PHYSMOL.

Solves the vocabulary bottleneck by:
  1. Loading pre-trained word vectors (fastText, Word2Vec, GloVe)
  2. Integrating sentence-transformers for context-aware encoding
  3. Supporting proper Chinese tokenization (jieba)
  4. Building a comprehensive vocabulary from multiple sources

Usage:
    encoder = EnhancedTextEncoder(vsa_dim=4096)

    # Load pre-trained vectors (one-time setup)
    encoder.load_pretrained_vectors("path/to/cc.zh.300.vec")

    # Or use sentence-transformers (recommended)
    encoder.init_sentence_transformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Encode text
    vec = encoder.encode("这是一个测试")
"""

from __future__ import annotations

import os
import re
import json
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path

import numpy as np


class PretrainedVectorLoader:
    """Load pre-trained word vectors from various formats."""

    @staticmethod
    def load_fasttext_vec(path: str, max_words: int = 100000,
                          encoding: str = 'utf-8') -> Dict[str, np.ndarray]:
        """Load fastText .vec file.

        Format:
            <num_words> <dim>
            word 0.1 0.2 0.3 ...
        """
        vectors = {}
        with open(path, 'r', encoding=encoding, errors='ignore') as f:
            first_line = f.readline().strip()
            parts = first_line.split()
            if len(parts) == 2:
                # Header line: num_words dim
                num_words = int(parts[0])
                dim = int(parts[1])
            else:
                # No header, first line is a word vector
                num_words = max_words
                dim = len(parts) - 1
                word = parts[0]
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                vectors[word] = vec

            for i, line in enumerate(f):
                if i >= max_words:
                    break
                parts = line.strip().split()
                if len(parts) < dim + 1:
                    continue
                word = parts[0]
                try:
                    vec = np.array([float(x) for x in parts[1:dim+1]], dtype=np.float32)
                    vectors[word] = vec
                except ValueError:
                    continue

        return vectors

    @staticmethod
    def load_word2vec_bin(path: str, max_words: int = 100000
                          ) -> Dict[str, np.ndarray]:
        """Load Word2Vec binary format."""
        try:
            from gensim.models import KeyedVectors
            model = KeyedVectors.load_word2vec_format(path, binary=True, limit=max_words)
            vectors = {}
            for word in model.index_to_key:
                vectors[word] = model[word].astype(np.float32)
            return vectors
        except ImportError:
            print("gensim not installed. Install with: pip install gensim")
            return {}

    @staticmethod
    def load_glove(path: str, max_words: int = 100000
                   ) -> Dict[str, np.ndarray]:
        """Load GloVe format."""
        vectors = {}
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i >= max_words:
                    break
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                word = parts[0]
                try:
                    vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    vectors[word] = vec
                except ValueError:
                    continue
        return vectors


class ChineseTokenizer:
    """Chinese tokenization with multiple backends."""

    def __init__(self):
        self._jieba = None
        self._thulac = None
        self._init_jieba()

    def _init_jieba(self):
        """Try to initialize jieba."""
        try:
            import jieba
            self._jieba = jieba
            # Disable jieba debug output
            jieba.setLogLevel(20)
        except ImportError:
            pass

    def tokenize(self, text: str) -> List[str]:
        """Tokenize Chinese text."""
        if self._jieba:
            return list(self._jieba.cut(text))
        else:
            # Fallback: character-level tokenization for Chinese
            tokens = []
            current_word = []
            for char in text:
                if '\u4e00' <= char <= '\u9fff':
                    # Chinese character
                    if current_word:
                        tokens.append(''.join(current_word))
                        current_word = []
                    tokens.append(char)
                elif char.isalnum():
                    current_word.append(char)
                else:
                    if current_word:
                        tokens.append(''.join(current_word))
                        current_word = []
            if current_word:
                tokens.append(''.join(current_word))
            return tokens


class EnhancedTextEncoder:
    """Enhanced text encoder with large vocabulary support.

    Features:
      - Pre-trained word vectors (fastText, Word2Vec, GloVe)
      - Sentence-transformers for context-aware encoding
      - Proper Chinese tokenization
      - Automatic vocabulary expansion
    """

    def __init__(self, vsa_dim: int = 4096, seed: int = 42):
        self.vsa_dim = vsa_dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Word vectors: {word: vector} (original dimension)
        self._word_vectors: Dict[str, np.ndarray] = {}

        # VSA projection: maps word vectors to VSA space
        self._projection: Optional[np.ndarray] = None
        self._vector_dim: int = 0

        # Sentence transformer
        self._st_model = None
        self._st_proj: Optional[np.ndarray] = None

        # Chinese tokenizer
        self._tokenizer = ChineseTokenizer()

        # Word-to-VSA fallback (for words not in pre-trained vectors)
        self._vsa_fallback: Dict[str, np.ndarray] = {}

        # Statistics
        self._stats = {
            "pretrained_words": 0,
            "fallback_words": 0,
            "total_encoded": 0,
        }

    # ------------------------------------------------------------------
    # Pre-trained vector loading
    # ------------------------------------------------------------------

    def load_pretrained_vectors(self, path: str, format: str = "auto",
                                max_words: int = 100000) -> int:
        """Load pre-trained word vectors.

        Args:
            path: path to vector file
            format: "fasttext", "word2vec", "glove", or "auto"
            max_words: maximum words to load

        Returns: number of words loaded
        """
        if format == "auto":
            if path.endswith(".bin"):
                format = "word2vec"
            elif path.endswith(".vec") or path.endswith(".txt"):
                # Check first line to determine format
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    first = f.readline()
                    if first.strip().split()[0].isdigit():
                        format = "fasttext"
                    else:
                        format = "glove"
            else:
                format = "fasttext"

        if format == "fasttext":
            vectors = PretrainedVectorLoader.load_fasttext_vec(path, max_words)
        elif format == "word2vec":
            vectors = PretrainedVectorLoader.load_word2vec_bin(path, max_words)
        elif format == "glove":
            vectors = PretrainedVectorLoader.load_glove(path, max_words)
        else:
            raise ValueError(f"Unknown format: {format}")

        self._word_vectors.update(vectors)
        self._stats["pretrained_words"] = len(self._word_vectors)

        # Determine vector dimension
        if vectors:
            sample_vec = next(iter(vectors.values()))
            self._vector_dim = len(sample_vec)
            self._build_projection()

        return len(vectors)

    def load_vocabulary(self, words: List[str]) -> int:
        """Load a vocabulary list (generates random vectors for unknown words)."""
        count = 0
        for word in words:
            if word not in self._word_vectors and word not in self._vsa_fallback:
                self._vsa_fallback[word] = self._rand_bipolar()
                count += 1
        self._stats["fallback_words"] = len(self._vsa_fallback)
        return count

    # ------------------------------------------------------------------
    # Sentence transformer
    # ------------------------------------------------------------------

    def init_sentence_transformer(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
                                   ) -> bool:
        """Initialize sentence-transformer for context-aware encoding.

        Recommended models:
          - paraphrase-multilingual-MiniLM-L12-v2 (multilingual, 384 dim)
          - all-MiniLM-L6-v2 (English only, 384 dim)
          - distiluse-base-multilingual-cased-v2 (multilingual, 512 dim)
        """
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(model_name)
            st_dim = self._st_model.get_sentence_embedding_dimension()

            # Build projection from ST space to VSA space
            self._st_proj = self.rng.randn(st_dim, self.vsa_dim).astype(np.float32)
            self._st_proj /= np.linalg.norm(self._st_proj, axis=1, keepdims=True)

            print(f"Loaded sentence-transformer: {model_name} ({st_dim} dim)")
            return True
        except ImportError:
            print("sentence-transformers not installed.")
            print("Install with: pip install sentence-transformers")
            return False
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """Encode text into VSA vector.

        Priority:
          1. Sentence-transformer (if available)
          2. Pre-trained word vectors + projection
          3. Fallback random vectors
        """
        self._stats["total_encoded"] += 1

        # Use sentence-transformer if available
        if self._st_model is not None:
            return self._encode_with_transformer(text)

        # Use word-level encoding
        tokens = self._tokenize(text)
        return self._encode_tokens(tokens)

    def encode_word(self, word: str) -> np.ndarray:
        """Encode a single word."""
        word_lower = word.lower()

        # Check pre-trained vectors
        if word_lower in self._word_vectors:
            vec = self._word_vectors[word_lower]
            if self._projection is not None:
                return vec @ self._projection
            return self._project_random(vec)

        # Check fallback
        if word_lower in self._vsa_fallback:
            return self._vsa_fallback[word_lower]

        # Generate new fallback vector
        vec = self._rand_bipolar()
        self._vsa_fallback[word_lower] = vec
        self._stats["fallback_words"] = len(self._vsa_fallback)
        return vec

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts efficiently."""
        if self._st_model is not None:
            embeddings = self._st_model.encode(texts, convert_to_numpy=True, batch_size=32)
            return (embeddings @ self._st_proj).astype(np.float32)
        else:
            return np.array([self.encode(t) for t in texts])

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text (supports Chinese and English)."""
        text = text.lower().strip()

        # Split by whitespace and punctuation
        tokens = []
        # Use Chinese tokenizer for Chinese text
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            tokens = self._tokenizer.tokenize(text)
        else:
            tokens = re.findall(r'[a-z]+|[0-9]+', text)

        return [t for t in tokens if len(t) > 0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_with_transformer(self, text: str) -> np.ndarray:
        """Encode using sentence-transformer."""
        embedding = self._st_model.encode(text, convert_to_numpy=True)
        vec = embedding @ self._st_proj
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32)

    def _encode_tokens(self, tokens: List[str]) -> np.ndarray:
        """Encode tokens using word vectors + positional binding."""
        if not tokens:
            return np.zeros(self.vsa_dim, dtype=np.float32)

        # Positional encoding vectors
        pos_vectors = [self._rand_bipolar() for _ in range(min(len(tokens), 64))]

        vec = np.zeros(self.vsa_dim, dtype=np.float32)
        for i, token in enumerate(tokens):
            word_vec = self.encode_word(token)
            pos_vec = pos_vectors[min(i, len(pos_vectors) - 1)]
            vec += word_vec * pos_vec

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _build_projection(self):
        """Build random projection from word vector space to VSA space."""
        if self._vector_dim > 0:
            self._projection = self.rng.randn(self._vector_dim, self.vsa_dim).astype(np.float32)
            self._projection /= np.linalg.norm(self._projection, axis=1, keepdims=True)

    def _project_random(self, vec: np.ndarray) -> np.ndarray:
        """Project a vector to VSA space using random projection."""
        if self._projection is not None:
            result = vec @ self._projection
        else:
            # Generate a deterministic mapping
            result = self._rand_bipolar()

        norm = np.linalg.norm(result)
        if norm > 0:
            result /= norm
        return result

    def _rand_bipolar(self) -> np.ndarray:
        """Generate a random bipolar vector."""
        return (self.rng.randint(0, 2, self.vsa_dim).astype(np.float32) * 2 - 1)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get encoder statistics."""
        return {
            "vsa_dim": self.vsa_dim,
            "vector_dim": self._vector_dim,
            "pretrained_words": self._stats["pretrained_words"],
            "fallback_words": self._stats["fallback_words"],
            "total_words": self._stats["pretrained_words"] + self._stats["fallback_words"],
            "total_encoded": self._stats["total_encoded"],
            "has_sentence_transformer": self._st_model is not None,
            "has_projection": self._projection is not None,
        }

    def save_vocabulary(self, path: str):
        """Save the vocabulary to a file."""
        data = {
            "vsa_dim": self.vsa_dim,
            "vector_dim": self._vector_dim,
            "fallback_words": {k: v.tolist() for k, v in self._vsa_fallback.items()},
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_vocabulary_file(self, path: str):
        """Load vocabulary from a file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.vsa_dim = data.get("vsa_dim", self.vsa_dim)
        self._vector_dim = data.get("vector_dim", self._vector_dim)

        for k, v in data.get("fallback_words", {}).items():
            self._vsa_fallback[k] = np.array(v, dtype=np.float32)

        self._stats["fallback_words"] = len(self._vsa_fallback)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def download_fasttext_vectors(lang: str = "zh", output_dir: str = "./vectors") -> str:
    """Download fastText pre-trained vectors.

    Args:
        lang: language code ("zh" for Chinese, "en" for English)
        output_dir: directory to save vectors

    Returns: path to downloaded file
    """
    import urllib.request
    import gzip
    import shutil

    os.makedirs(output_dir, exist_ok=True)

    urls = {
        "zh": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz",
        "en": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz",
        "ja": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz",
        "ko": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.vec.gz",
    }

    if lang not in urls:
        raise ValueError(f"Unsupported language: {lang}. Choose from: {list(urls.keys())}")

    url = urls[lang]
    gz_path = os.path.join(output_dir, f"cc.{lang}.300.vec.gz")
    vec_path = os.path.join(output_dir, f"cc.{lang}.300.vec")

    if os.path.exists(vec_path):
        print(f"Vectors already exist: {vec_path}")
        return vec_path

    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, gz_path)

    print(f"Extracting to {vec_path}...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(vec_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(gz_path)
    print(f"Done: {vec_path}")
    return vec_path
