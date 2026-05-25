"""PHYSMOL Dual-Helix Cross-Modal Alignment (InfoNCE)."""

import numpy as np
from typing import Optional, Tuple

try:
    from . import _vsa
    from .vsa import VectorSymbolicArchitecture, VSAVector
except ImportError:
    _vsa = None


class AlignmentHub:
    """InfoNCE-based alignment between physical VSA space and language embeddings.

    Implements Eq.21-22 from PHYSMOL paper:
    L_align = -log(exp(v_phys · v_lang / τ) / Σ_i exp(v_phys · v_i / τ))
    """

    def __init__(self, vsa_dim: int = 4096, lang_dim: int = 300,
                 temperature: float = 0.07):
        self.vsa_dim = vsa_dim
        self.lang_dim = lang_dim
        self.temperature = temperature

        # Projection matrix: lang_dim -> vsa_dim
        # Maps word vectors into VSA space
        self.W_proj = np.random.randn(lang_dim, vsa_dim).astype(np.float32)
        self.W_proj /= np.linalg.norm(self.W_proj, axis=1, keepdims=True)

        # Bias
        self.b_proj = np.zeros(vsa_dim, dtype=np.float32)

    def project(self, lang_vec: np.ndarray) -> np.ndarray:
        """Project a language vector into VSA space."""
        return lang_vec @ self.W_proj + self.b_proj

    def info_nce_loss(self, v_phys: np.ndarray, v_lang_pos: np.ndarray,
                      v_lang_negatives: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute InfoNCE loss and gradients.

        Args:
            v_phys: (vsa_dim,) physical VSA vector
            v_lang_pos: (vsa_dim,) projected positive language vector
            v_lang_negatives: (N, vsa_dim) projected negative language vectors

        Returns:
            loss: scalar
            grad_v_phys: gradient w.r.t. v_phys
        """
        tau = self.temperature

        # Positive similarity
        s_pos = np.dot(v_phys, v_lang_pos) / tau

        # Negative similarities
        s_neg = v_lang_negatives @ v_phys / tau

        # Log-sum-exp for numerical stability
        all_sims = np.concatenate([[s_pos], s_neg])
        max_s = np.max(all_sims)
        lse = max_s + np.log(np.sum(np.exp(all_sims - max_s)))

        # Loss
        loss = lse - s_pos

        # Gradient: ∂L/∂v_phys = (1/τ) Σ_{i≠+} p_i (v_i - v_lang_pos)
        # where p_i = exp(s_i) / Σ_j exp(s_j)
        exp_sims = np.exp(all_sims - max_s)
        probs = exp_sims / np.sum(exp_sims)

        grad = np.zeros_like(v_phys)
        # prob[0] is positive, probs[1:] are negatives
        for i, neg in enumerate(v_lang_negatives):
            grad += probs[i + 1] * (neg - v_lang_pos)
        grad /= tau

        return loss, grad

    def update_projection(self, lang_vecs: np.ndarray, vsa_vecs: np.ndarray,
                          lr: float = 0.001):
        """Update projection matrix to align language and VSA spaces.

        Uses orthogonal regularization to preserve semantic topology.
        """
        # Simple gradient step on W_proj
        # Minimize ||project(lang) - vsa||^2
        projected = lang_vecs @ self.W_proj + self.b_proj
        error = projected - vsa_vecs

        # Gradient for W_proj: lang_vecs.T @ error
        grad_W = lang_vecs.T @ error / len(lang_vecs)
        grad_b = np.mean(error, axis=0)

        self.W_proj -= lr * grad_W
        self.b_proj -= lr * grad_b

        # Orthogonal regularization (lightweight)
        # Encourage W_proj rows to be approximately orthogonal
        WtW = self.W_proj @ self.W_proj.T
        reg_grad = 2.0 * (WtW - np.eye(self.lang_dim)) @ self.W_proj
        self.W_proj -= lr * 0.01 * reg_grad

    def align_batch(self, phys_vecs: np.ndarray, lang_vecs: np.ndarray,
                    negative_samples: int = 128) -> float:
        """Run one alignment step on a batch. Returns average loss."""
        total_loss = 0.0
        for i in range(len(phys_vecs)):
            v_phys = phys_vecs[i]
            v_lang_pos = self.project(lang_vecs[i])

            # Sample negatives from other items
            neg_indices = np.random.choice(
                len(lang_vecs), size=min(negative_samples, len(lang_vecs) - 1),
                replace=False
            )
            # Remove positive index
            neg_indices = neg_indices[neg_indices != i]
            v_lang_neg = self.project(lang_vecs[neg_indices])

            loss, grad = self.info_nce_loss(v_phys, v_lang_pos, v_lang_neg)
            total_loss += loss

        return total_loss / len(phys_vecs)
