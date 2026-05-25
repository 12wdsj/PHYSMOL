"""PHYSMOL Serial Two-Phase Training.

Phase 1 (Physical): MuJoCo exploration -> multimodal perception -> VSA concepts -> LNN/SNN
Phase 2 (Language):  Load pre-trained word vectors -> align with Phase 1 VSA space

Why serial instead of dual-helix:
- 23-core CPU is the bottleneck for MuJoCo simulation (CPU-bound)
- 192GB GPU sits idle during physical spiral
- Serial approach: Phase 1 saturates CPU, Phase 2 saturates GPU
- Result: better hardware utilization than parallel dual-helix
"""

import numpy as np
import time
import os
import json
from typing import Optional, Dict, Any, List

from .lnn import LagrangianNetwork
from .alignment import AlignmentHub
from .world_model import HierarchicalWorldModel
from .perception import MultiModalPerception

# Conditional imports for C-extension-dependent modules
try:
    from .vsa import VectorSymbolicArchitecture, Codebook, FHRRSpace
    from .snn import SpikingNetwork, CausalGraph, ThreeFactorLearner
    _HAS_C = True
except ImportError:
    _HAS_C = False
    # Stub classes for when C extensions aren't built
    class _Stub:
        def __init__(self, *a, **kw): pass
        def __getattr__(self, name): raise ImportError("C extensions not built. Run: make build")
    VectorSymbolicArchitecture = Codebook = FHRRSpace = _Stub
    SpikingNetwork = CausalGraph = ThreeFactorLearner = _Stub


class PhysicalPhase:
    """Phase 1: Physical exploration in MuJoCo.

    Generates interaction data and builds VSA concept vectors.
    CPU-bound: MuJoCo simulation is the bottleneck.
    """

    def __init__(self, config: Dict[str, Any]):
        self.vsa_dim = config.get('vsa_dim', 4096)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.coord_dim = config.get('coord_dim', 6)
        self.num_episodes = config.get('num_episodes', 1000)
        self.steps_per_episode = config.get('steps_per_episode', 500)
        self.lr = config.get('lr', 0.001)

        # Core components
        self.vsa = VectorSymbolicArchitecture(self.vsa_dim)
        self.codebook = Codebook(self.vsa_dim)
        self.fhrr = FHRRSpace(self.vsa_dim)
        self.lnn = LagrangianNetwork(self.coord_dim, self.hidden_dim)
        self.world_model = HierarchicalWorldModel(self.coord_dim)
        self.perception = MultiModalPerception(self.vsa_dim)

        # SNN for causal reasoning
        snn_dim = min(self.vsa_dim // 8, 512)
        self.snn = SpikingNetwork(snn_dim, snn_dim)
        self.causal = CausalGraph(max_nodes=1024)

        # Data storage for Phase 2
        self.object_vectors = {}  # name -> VSA vector (numpy)
        self.experience_buffer = []  # (state, action, next_state, reward, perception)
        self.lnn_losses = []

    def collect_experience(self, env, num_steps: int = 500) -> List[Dict]:
        """Run one episode and collect multimodal experience."""
        env.reset()
        episode_data = []

        for t in range(num_steps):
            # Random action for exploration
            action = np.random.randn(self.coord_dim).astype(np.float32) * 0.1

            state_before = env.get_state()
            env.step(action)
            state_after = env.get_state()

            # Multimodal perception of all objects
            perceptions = env.get_all_object_perceptions()

            # Store experience
            episode_data.append({
                'step': t,
                'state': state_before['qpos'].copy(),
                'action': action,
                'next_state': state_after['qpos'].copy(),
                'perceptions': perceptions,
                'collisions': list(env._collision_log[-10:]) if hasattr(env, '_collision_log') else [],
            })

            # Register new objects in codebook
            for name, vec in perceptions.items():
                if name not in self.object_vectors:
                    self.object_vectors[name] = vec

        return episode_data

    def learn_from_experience(self, episode_data: List[Dict]) -> Dict[str, float]:
        """Learn from collected experience: LNN dynamics + SNN causality."""
        metrics = {'lnn_loss': 0.0, 'causal_edges': 0, 'steps': 0}

        for exp in episode_data:
            state = exp['state']
            action = exp['action']
            next_state = exp['next_state']

            if len(state) < self.coord_dim or len(next_state) < self.coord_dim:
                continue

            q = state[:self.coord_dim].astype(np.float32)
            q_next = next_state[:self.coord_dim].astype(np.float32)
            q_dot = (q_next - q) / 0.01  # dt=0.01

            # LNN: predict acceleration, compare with observed
            try:
                predicted_ddot = self.lnn.compute_acceleration(q, q_dot)
                observed_ddot = q_dot / 0.01
                loss = float(np.mean((predicted_ddot - observed_ddot) ** 2))
                metrics['lnn_loss'] += loss
            except Exception:
                pass

            # SNN: encode temporal relationship between states
            # Use perception vectors as spike input (threshold)
            for name, percept in exp.get('perceptions', {}).items():
                # Downsample perception to SNN input size
                snn_input = percept[:self.snn.num_pre]
                if len(snn_input) < self.snn.num_pre:
                    snn_input = np.pad(snn_input, (0, self.snn.num_pre - len(snn_input)))
                spike_input = (np.abs(snn_input) > np.median(np.abs(snn_input))).astype(np.float32)

                pre_spikes, post_spikes = self.snn.step(spike_input)
                self.snn.stdp(a_plus=0.005, a_minus=0.006)

            # Causal graph: connect collision events
            for collision in exp.get('collisions', []):
                g1, g2 = collision['geom1'], collision['geom2']
                force = np.linalg.norm(collision['force'])
                self.causal.add_edge(g1, g2, weight=force, credit=force * 0.1)

            metrics['steps'] += 1

        # Average LNN loss
        if metrics['steps'] > 0:
            metrics['lnn_loss'] /= metrics['steps']
        metrics['causal_edges'] = self.causal.edge_count

        return metrics

    def build_concept_vectors(self) -> Dict[str, np.ndarray]:
        """Build final VSA concept vectors for all discovered objects.

        Each concept = bundling of all modality-specific vectors.
        """
        return dict(self.object_vectors)

    def run(self, env, progress_callback=None) -> Dict[str, Any]:
        """Run full Phase 1: physical exploration and learning."""
        print(f"[Phase 1] Physical exploration: {self.num_episodes} episodes, "
              f"{self.steps_per_episode} steps each")

        all_metrics = []
        start_time = time.time()

        for ep in range(self.num_episodes):
            # Collect experience
            episode_data = self.collect_experience(env, self.steps_per_episode)

            # Learn from experience
            metrics = self.learn_from_experience(episode_data)
            self.lnn_losses.append(metrics['lnn_loss'])
            all_metrics.append(metrics)

            if progress_callback:
                progress_callback(ep, self.num_episodes, metrics)

            # Print progress every 100 episodes
            if (ep + 1) % 100 == 0:
                elapsed = time.time() - start_time
                eps_per_sec = (ep + 1) / elapsed
                eta = (self.num_episodes - ep - 1) / eps_per_sec
                print(f"  Episode {ep+1}/{self.num_episodes} | "
                      f"LNN loss: {metrics['lnn_loss']:.6f} | "
                      f"Causal edges: {metrics['causal_edges']} | "
                      f"ETA: {eta:.0f}s")

        elapsed = time.time() - start_time
        concepts = self.build_concept_vectors()

        print(f"[Phase 1] Complete in {elapsed:.1f}s")
        print(f"  Discovered {len(concepts)} object concepts")
        print(f"  Final LNN loss: {self.lnn_losses[-1]:.6f}")
        print(f"  Causal graph: {self.causal.edge_count} edges")

        return {
            'concepts': concepts,
            'lnn_losses': self.lnn_losses,
            'causal_edges': self.causal.edge_count,
            'elapsed': elapsed,
        }


class LanguagePhase:
    """Phase 2: Language alignment with pre-built physical concepts.

    GPU-friendly: matrix operations on word vectors and VSA space.
    Runs after Phase 1 completes.
    """

    def __init__(self, config: Dict[str, Any]):
        self.vsa_dim = config.get('vsa_dim', 4096)
        self.lang_dim = config.get('lang_dim', 300)
        self.temperature = config.get('temperature', 0.07)
        self.num_epochs = config.get('lang_epochs', 50)
        self.lr = config.get('lr', 0.001)
        self.batch_size = config.get('batch_size', 64)

        self.alignment = AlignmentHub(self.vsa_dim, self.lang_dim, self.temperature)

    def load_word_vectors(self, path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Load pre-trained word vectors.

        If no path provided, generates simple synthetic word vectors
        for testing (not for production use).
        """
        if path and os.path.exists(path):
            # Load fastText/GloVe format
            return self._load_text_format(path)

        # Synthetic word vectors for testing
        print("  WARNING: Using synthetic word vectors. "
              "For production, provide fastText/GloVe vectors.")
        words = [
            "red", "blue", "green", "sphere", "cube", "cylinder",
            "heavy", "light", "hard", "soft", "metal", "rubber", "wood",
            "ball", "block", "drop", "roll", "bounce", "collide", "slide",
            "fast", "slow", "smooth", "rough", "round", "square",
        ]
        rng = np.random.RandomState(42)
        word_vectors = {}
        for w in words:
            word_vectors[w] = rng.randn(self.lang_dim).astype(np.float32)
            word_vectors[w] /= np.linalg.norm(word_vectors[w])
        return word_vectors

    def _load_text_format(self, path: str) -> Dict[str, np.ndarray]:
        """Load word vectors in text format (word dim1 dim2 ...)."""
        vectors = {}
        with open(path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            parts = first_line.strip().split()
            # Check if first line is header (num_words dim)
            if len(parts) == 2 and parts[0].isdigit():
                pass  # skip header
            else:
                f.seek(0)

            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                word = parts[0]
                try:
                    vec = np.array([float(x) for x in parts[1:self.lang_dim+1]],
                                   dtype=np.float32)
                    if len(vec) == self.lang_dim:
                        vectors[word] = vec
                except ValueError:
                    continue
        return vectors

    def create_alignment_pairs(self, concepts: Dict[str, np.ndarray],
                                word_vectors: Dict[str, np.ndarray]
                                ) -> tuple:
        """Create (physical_vec, language_vec) pairs for alignment.

        Maps object concepts to word labels using simple heuristics.
        """
        # Simple label mapping (in production, this would come from
        # the physical spiral's language input or human annotations)
        label_map = {
            'red_ball': ['red', 'ball', 'sphere', 'round'],
            'blue_cube': ['blue', 'cube', 'block', 'square'],
            'green_cyl': ['green', 'cylinder', 'wood'],
        }

        phys_vecs = []
        lang_vecs = []
        labels = []

        for obj_name, concept_vec in concepts.items():
            words = label_map.get(obj_name, [obj_name])
            for word in words:
                if word in word_vectors:
                    phys_vecs.append(concept_vec)
                    lang_vecs.append(word_vectors[word])
                    labels.append(f"{obj_name} <-> {word}")

        if not phys_vecs:
            return np.zeros((0, self.vsa_dim)), np.zeros((0, self.lang_dim)), []

        return np.array(phys_vecs), np.array(lang_vecs), labels

    def run(self, concepts: Dict[str, np.ndarray],
            word_vectors_path: Optional[str] = None) -> Dict[str, Any]:
        """Run Phase 2: language alignment."""
        print(f"[Phase 2] Language alignment: {self.num_epochs} epochs")

        word_vectors = self.load_word_vectors(word_vectors_path)
        print(f"  Loaded {len(word_vectors)} word vectors")

        phys_vecs, lang_vecs, labels = self.create_alignment_pairs(
            concepts, word_vectors)

        if len(phys_vecs) == 0:
            print("  WARNING: No alignment pairs found. Skipping.")
            return {'losses': [], 'pairs': 0}

        print(f"  Created {len(phys_vecs)} alignment pairs")

        losses = []
        start_time = time.time()

        for epoch in range(self.num_epochs):
            # Shuffle
            indices = np.random.permutation(len(phys_vecs))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(indices), self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                p_batch = phys_vecs[batch_idx]
                l_batch = lang_vecs[batch_idx]

                loss = self.alignment.align_batch(p_batch, l_batch)
                self.alignment.update_projection(l_batch, p_batch, lr=self.lr)
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.num_epochs} | Loss: {avg_loss:.4f}")

        elapsed = time.time() - start_time
        print(f"[Phase 2] Complete in {elapsed:.1f}s | Final loss: {losses[-1]:.4f}")

        return {
            'losses': losses,
            'pairs': len(phys_vecs),
            'elapsed': elapsed,
            'word_vectors': word_vectors,
        }


class SerialTrainer:
    """Two-phase serial training: Physical first, then Language.

    Phase 1: CPU-bound (MuJoCo simulation + LNN/SNN learning)
    Phase 2: GPU-bound (matrix operations for alignment)

    This is more efficient than dual-helix on hardware where
    CPU is the bottleneck (23 cores) and GPU is underutilized (192GB).
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.config.setdefault('vsa_dim', 4096)
        self.config.setdefault('hidden_dim', 128)
        self.config.setdefault('coord_dim', 6)
        self.config.setdefault('num_episodes', 1000)
        self.config.setdefault('steps_per_episode', 500)
        self.config.setdefault('lang_epochs', 50)
        self.config.setdefault('lr', 0.001)

        self.phase1 = PhysicalPhase(self.config)
        self.phase2 = LanguagePhase(self.config)

        self.results = {}

    def run_phase1(self, env) -> Dict[str, Any]:
        """Run Phase 1: Physical exploration."""
        self.results['phase1'] = self.phase1.run(env)
        return self.results['phase1']

    def run_phase2(self, word_vectors_path: Optional[str] = None) -> Dict[str, Any]:
        """Run Phase 2: Language alignment (requires Phase 1 concepts)."""
        if 'phase1' not in self.results:
            raise RuntimeError("Must run Phase 1 before Phase 2")

        concepts = self.results['phase1']['concepts']
        self.results['phase2'] = self.phase2.run(concepts, word_vectors_path)
        return self.results['phase2']

    def run_full(self, env, word_vectors_path: Optional[str] = None) -> Dict[str, Any]:
        """Run both phases sequentially."""
        print("=" * 60)
        print("PHYSMOL Serial Training")
        print(f"  VSA dim: {self.config['vsa_dim']}")
        print(f"  LNN hidden: {self.config['hidden_dim']}")
        print(f"  Episodes: {self.config['num_episodes']}")
        print("=" * 60)

        # Phase 1
        self.run_phase1(env)

        # Phase 2
        self.run_phase2(word_vectors_path)

        # Summary
        total_time = (self.results['phase1']['elapsed'] +
                      self.results['phase2']['elapsed'])
        print("=" * 60)
        print(f"Training complete in {total_time:.1f}s")
        print(f"  Phase 1 (physical): {self.results['phase1']['elapsed']:.1f}s")
        print(f"  Phase 2 (language): {self.results['phase2']['elapsed']:.1f}s")
        print(f"  Object concepts: {len(self.results['phase1']['concepts'])}")
        print(f"  Causal edges: {self.results['phase1']['causal_edges']}")
        print(f"  Alignment loss: {self.results['phase2']['losses'][-1]:.4f}")
        print("=" * 60)

        return self.results

    def save(self, path: str):
        """Save trained components to disk."""
        os.makedirs(path, exist_ok=True)

        # Save object concepts
        concepts = self.results.get('phase1', {}).get('concepts', {})
        for name, vec in concepts.items():
            np.save(os.path.join(path, f"concept_{name}.npy"), vec)

        # Save LNN losses
        losses = self.results.get('phase1', {}).get('lnn_losses', [])
        np.save(os.path.join(path, "lnn_losses.npy"), np.array(losses))

        # Save alignment losses
        align_losses = self.results.get('phase2', {}).get('losses', [])
        np.save(os.path.join(path, "alignment_losses.npy"), np.array(align_losses))

        # Save config
        with open(os.path.join(path, "config.json"), 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"Model saved to {path}")

    def get_status(self) -> Dict[str, Any]:
        return {
            'config': self.config,
            'phase1_done': 'phase1' in self.results,
            'phase2_done': 'phase2' in self.results,
            'world_model': self.phase1.world_model.get_status(),
        }


def create_trainer(vsa_dim: int = 4096, num_episodes: int = 1000) -> SerialTrainer:
    """Create a trainer with default configuration."""
    config = {
        'vsa_dim': vsa_dim,
        'hidden_dim': 128,
        'coord_dim': 6,
        'num_episodes': num_episodes,
        'steps_per_episode': 500,
        'lang_epochs': 50,
        'lr': 0.001,
        'temperature': 0.07,
        'batch_size': 64,
    }
    return SerialTrainer(config)
