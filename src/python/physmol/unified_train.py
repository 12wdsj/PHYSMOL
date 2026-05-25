"""PHYSMOL Unified Training Pipeline.

4-phase serial training:
  Phase 1: Physics Learning (GPU)     -- LGNN learns dynamics from analytical ground truth
  Phase 2: Concept Formation (CPU)    -- Extract attributes, register VSA recipes
  Phase 3: Language Alignment (GPU)   -- InfoNCE: text vectors <-> physical concept vectors
  Phase 4: End-to-End Integration     -- Wire LGNN into CognitiveInterface, validate

Usage:
    # Full pipeline
    python -m physmol.unified_train --device cuda --epochs 500 --save-path ./checkpoints

    # Individual phases
    python -m physmol.unified_train --phase 1 --device cuda --epochs 500
    python -m physmol.unified_train --phase 2
    python -m physmol.unified_train --phase 3 --device cuda --epochs 100
    python -m physmol.unified_train --phase 4 --checkpoint ./checkpoints
"""

import numpy as np
import time
import os
import json
from typing import Dict, List, Optional, Any, Tuple

from .lgnn import LagrangianGraphNetwork, PhysicsGraph, SpringMassSystem, GravitationalTwoBody
from .lgnn_train import LGNNTrainer, TrajectoryDataset
from .vsa_store import AttributePrimitivePool, RecipeStore, ConceptSynthesizer
from .language.text_encoder import TextToVSA
from .language.cognitive import CognitiveInterface


# ---------------------------------------------------------------------------
# Phase 1: Physics Learning
# ---------------------------------------------------------------------------

class Phase1Physics:
    """Train LGNN on analytical physics (spring-mass + gravity)."""

    def __init__(self, coord_dim: int = 2, hidden_dim: int = 64,
                 num_layers: int = 3, seed: int = 42):
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seed = seed

    def run(self, device: str = 'auto', epochs: int = 500,
            batch_size: int = 32, save_path: Optional[str] = None) -> Dict:
        """Run Phase 1: train LGNN."""
        print("=" * 60)
        print("Phase 1: Physics Learning (LGNN)")
        print("=" * 60)

        if save_path:
            os.makedirs(save_path, exist_ok=True)

        trainer = LGNNTrainer(
            coord_dim=self.coord_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            device=device,
            seed=self.seed,
        )

        results = trainer.train(
            epochs=epochs,
            batch_size=batch_size,
            save_path=save_path,
        )

        # Also generate evaluation trajectories for Phase 2
        dataset = TrajectoryDataset(self.coord_dim, self.seed)
        spring_data = dataset.generate_spring_batch(n_trajectories=20, steps=50)
        gravity_data = dataset.generate_gravity_batch(n_trajectories=20, steps=50)

        results['trainer'] = trainer
        results['spring_data'] = spring_data
        results['gravity_data'] = gravity_data

        return results


# ---------------------------------------------------------------------------
# Phase 2: Concept Formation
# ---------------------------------------------------------------------------

class Phase2Concepts:
    """Extract physical attributes and register VSA recipes."""

    def __init__(self, vsa_dim: int = 10000, seed: int = 42):
        self.vsa_dim = vsa_dim
        self.seed = seed

    def _classify_mass(self, mass: float) -> str:
        """Classify mass value into attribute ID."""
        if mass < 0.7:
            return "mass_very_light"
        elif mass < 1.2:
            return "mass_light"
        elif mass < 1.8:
            return "mass_medium"
        elif mass < 2.5:
            return "mass_heavy"
        else:
            return "mass_very_heavy"

    def _classify_elasticity(self, k: float) -> str:
        """Classify spring constant into elasticity attribute."""
        if k < 3.0:
            return "elasticity_soft"
        elif k < 6.0:
            return "elasticity_elastic"
        elif k < 10.0:
            return "elasticity_stiff"
        else:
            return "elasticity_rigid"

    def _generate_object_scenarios(self, n_scenarios: int = 50
                                   ) -> List[Dict[str, Any]]:
        """Generate diverse object scenarios for concept formation."""
        rng = np.random.RandomState(self.seed)
        scenarios = []

        shapes = ["sphere", "cube", "cylinder"]
        colors = ["red", "blue", "green", "yellow", "white"]
        materials = ["rubber", "metal", "wood", "plastic"]
        textures = ["smooth", "rough"]

        for i in range(n_scenarios):
            shape = rng.choice(shapes)
            color = rng.choice(colors)
            material = rng.choice(materials)
            texture = rng.choice(textures)
            mass = rng.uniform(0.3, 3.0)
            k = rng.uniform(2.0, 15.0)

            scenarios.append({
                'id': f"obj_{i:03d}",
                'shape': shape,
                'color': color,
                'material': material,
                'texture': texture,
                'mass': mass,
                'spring_k': k,
            })

        return scenarios

    def run(self, n_scenarios: int = 50, save_path: Optional[str] = None) -> Dict:
        """Run Phase 2: concept formation."""
        print("=" * 60)
        print("Phase 2: Concept Formation (VSA Recipes)")
        print("=" * 60)

        primitives = AttributePrimitivePool(self.vsa_dim, self.seed)
        store = RecipeStore(primitives)
        synthesizer = ConceptSynthesizer(store)

        # Generate scenarios
        scenarios = self._generate_object_scenarios(n_scenarios)
        print(f"  Generated {len(scenarios)} object scenarios")

        # Register each object as a recipe
        for obj in scenarios:
            attr_ids = [
                f"shape_{obj['shape']}",
                f"color_{obj['color']}",
                f"material_{obj['material']}",
                f"texture_{obj['texture']}",
                self._classify_mass(obj['mass']),
                self._classify_elasticity(obj['spring_k']),
            ]
            store.register_recipe(obj['id'], attr_ids)

        print(f"  Registered {len(store)} recipes")

        # Test resonance search
        print("  Testing resonance search...")
        test_vec = synthesizer.compose_concept({
            'shape': 'sphere', 'color': 'red', 'material': 'rubber'
        })
        matches = store.resonate(test_vec, top_k=3)
        for obj_id, sim in matches:
            print(f"    {obj_id}: similarity={sim:.4f}")

        # Test decomposition
        print("  Testing decomposition...")
        decomp = store.decompose(test_vec)
        for cat, (name, conf) in decomp.items():
            print(f"    {cat}: {name} (confidence={conf:.4f})")

        # Save
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            # Save recipes
            recipes_data = {
                obj_id: store.get_recipe(obj_id)
                for obj_id in store.list_objects()
            }
            with open(os.path.join(save_path, 'recipes.json'), 'w') as f:
                json.dump(recipes_data, f, indent=2)
            print(f"  Saved recipes to {save_path}")

        return {
            'primitives': primitives,
            'store': store,
            'synthesizer': synthesizer,
            'scenarios': scenarios,
            'num_recipes': len(store),
        }


# ---------------------------------------------------------------------------
# Phase 3: Language Alignment
# ---------------------------------------------------------------------------

class Phase3Language:
    """Align language vectors with physical concept vectors via InfoNCE."""

    def __init__(self, vsa_dim: int = 10000, seed: int = 42):
        self.vsa_dim = vsa_dim
        self.seed = seed

    def _create_alignment_pairs(self, store: RecipeStore,
                                text_encoder: TextToVSA,
                                scenarios: List[Dict]
                                ) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        """Create (physical_vec, text_vec) pairs for alignment."""
        phys_vecs = []
        text_vecs = []
        labels = []

        # Create natural language descriptions for each object
        for obj in scenarios:
            # Physical vector: synthesize from recipe
            phys_vec = store.synthesize(obj['id'])
            if phys_vec is None:
                continue

            # Text description
            desc = (f"a {obj['color']} {obj['material']} {obj['shape']} "
                    f"that is {obj['texture']}")
            text_vec = text_encoder.encode(desc)

            phys_vecs.append(phys_vec)
            text_vecs.append(text_vec)
            labels.append(f"{obj['id']}: {desc}")

        return phys_vecs, text_vecs, labels

    def run(self, store: RecipeStore, scenarios: List[Dict],
            epochs: int = 100, lr: float = 0.001,
            save_path: Optional[str] = None) -> Dict:
        """Run Phase 3: language alignment."""
        print("=" * 60)
        print("Phase 3: Language Alignment")
        print("=" * 60)

        text_encoder = TextToVSA(self.vsa_dim, self.seed)

        # Create alignment pairs
        phys_vecs, text_vecs, labels = self._create_alignment_pairs(
            store, text_encoder, scenarios)
        print(f"  Created {len(phys_vecs)} alignment pairs")

        if len(phys_vecs) == 0:
            print("  WARNING: No alignment pairs. Skipping.")
            return {'losses': [], 'text_encoder': text_encoder}

        phys_arr = np.array(phys_vecs, dtype=np.float32)
        text_arr = np.array(text_vecs, dtype=np.float32)

        # Alignment training via gradient descent on projection
        # We optimize text_encoder word vectors to align with physical space
        losses = []
        rng = np.random.RandomState(self.seed)

        print(f"  Training for {epochs} epochs...")
        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = 0.0

            # Shuffle indices
            indices = rng.permutation(len(phys_vecs))

            for idx in indices:
                phys = phys_arr[idx]
                text = text_arr[idx]

                # Compute cosine similarity
                p_norm = np.linalg.norm(phys)
                t_norm = np.linalg.norm(text)
                if p_norm < 1e-10 or t_norm < 1e-10:
                    continue

                # InfoNCE-style loss
                # Positive pair similarity
                s_pos = np.dot(phys, text) / (p_norm * t_norm)

                # Negative pairs: random other texts
                neg_indices = rng.choice(
                    len(text_arr),
                    size=min(32, len(text_arr) - 1),
                    replace=False)
                neg_indices = neg_indices[neg_indices != idx]

                s_neg = []
                for neg_idx in neg_indices:
                    neg_text = text_arr[neg_idx]
                    neg_norm = np.linalg.norm(neg_text)
                    if neg_norm > 1e-10:
                        s = np.dot(phys, neg_text) / (p_norm * neg_norm)
                        s_neg.append(s)

                if not s_neg:
                    continue

                # Softmax
                all_sims = np.array([s_pos] + s_neg)
                tau = 0.07
                exp_sims = np.exp(all_sims / tau)
                probs = exp_sims / np.sum(exp_sims)

                # Loss: -log(p_pos)
                loss = -np.log(max(probs[0], 1e-10))
                epoch_loss += loss

                # Gradient update on word vectors
                # For each word in the description, nudge it toward physical vec
                desc = labels[idx].split(': ', 1)[1] if ': ' in labels[idx] else ""
                words = text_encoder.tokenize(desc)

                for word in words:
                    word_vec = text_encoder.lexicon.get_vector(word)
                    # Simple gradient step: move word vec toward physical vec
                    direction = phys / p_norm - word_vec
                    word_vec += lr * direction
                    # Re-normalize
                    w_norm = np.linalg.norm(word_vec)
                    if w_norm > 0:
                        word_vec /= w_norm
                    text_encoder.lexicon.set_vector(word, word_vec)

                # Re-encode text with updated vectors
                new_text_vec = text_encoder.encode(desc)
                text_arr[idx] = new_text_vec

            avg_loss = epoch_loss / len(indices)
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                      f"Time: {elapsed:.1f}s")

        total_time = time.time() - start_time
        print(f"  Alignment complete in {total_time:.1f}s")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Vocabulary size: {text_encoder.lexicon.vocabulary_size}")

        # Verify alignment
        print("  Verifying alignment...")
        test_desc = "a red rubber sphere"
        test_vec = text_encoder.encode(test_desc)
        matches = store.resonate(test_vec, top_k=3)
        print(f"    Query: '{test_desc}'")
        for obj_id, sim in matches:
            attrs = store.get_recipe(obj_id)
            print(f"    {obj_id}: sim={sim:.4f}")

        results = {
            'text_encoder': text_encoder,
            'losses': losses,
            'num_pairs': len(phys_vecs),
            'total_time': total_time,
        }

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            # Save word vectors
            word_vecs = {}
            for word in text_encoder.lexicon.words():
                word_vecs[word] = text_encoder.lexicon.get_vector(word).tolist()
            with open(os.path.join(save_path, 'word_vectors.json'), 'w') as f:
                json.dump({
                    'vsa_dim': self.vsa_dim,
                    'vocabulary': word_vecs,
                }, f)
            print(f"  Saved word vectors to {save_path}")

        return results


# ---------------------------------------------------------------------------
# Phase 4: End-to-End Integration
# ---------------------------------------------------------------------------

class Phase4Integration:
    """Wire everything together and validate the full pipeline."""

    def __init__(self, vsa_dim: int = 10000, seed: int = 42):
        self.vsa_dim = vsa_dim
        self.seed = seed

    def run(self, lgnn: Optional[LagrangianGraphNetwork] = None,
            store: Optional[RecipeStore] = None,
            text_encoder: Optional[TextToVSA] = None,
            scenarios: Optional[List[Dict]] = None,
            save_path: Optional[str] = None) -> Dict:
        """Run Phase 4: end-to-end integration."""
        print("=" * 60)
        print("Phase 4: End-to-End Integration")
        print("=" * 60)

        # Create CognitiveInterface
        ci = CognitiveInterface(vsa_dim=self.vsa_dim, seed=self.seed, lgnn=lgnn)

        # Register objects from scenarios
        if scenarios:
            for obj in scenarios:
                attr_ids = [
                    f"shape_{obj['shape']}",
                    f"color_{obj['color']}",
                    f"material_{obj['material']}",
                    f"texture_{obj['texture']}",
                ]
                # Add mass classification
                mass = obj.get('mass', 1.0)
                if mass < 0.7:
                    attr_ids.append("mass_very_light")
                elif mass < 1.2:
                    attr_ids.append("mass_light")
                elif mass < 1.8:
                    attr_ids.append("mass_medium")
                else:
                    attr_ids.append("mass_heavy")

                ci.register_object(obj['id'], attr_ids)

        print(f"  Registered {len(ci.list_objects())} objects")

        # Validation queries
        test_queries = [
            # Physics questions
            "What happens if I drop the red ball?",
            "What if the ball was heavier?",
            # Concept explanations
            "Explain elasticity",
            "Explain gravity",
            # Object queries
            "Tell me about the blue metal cube",
            # Commands
            "Push the sphere to the left",
            # Counterfactuals
            "What if there was no friction?",
        ]

        print(f"  Running {len(test_queries)} validation queries...")
        results = []
        for query in test_queries:
            response = ci.query(query)
            results.append({
                'query': query,
                'response': response,
            })
            print(f"    Q: {query}")
            print(f"    A: {response[:100]}...")
            print()

        # System status
        status = ci.get_status()
        print(f"  System status:")
        for key, val in status.items():
            print(f"    {key}: {val}")

        # Save
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, 'validation.json'), 'w') as f:
                json.dump({
                    'queries': results,
                    'status': status,
                }, f, indent=2)
            print(f"  Saved validation results to {save_path}")

        return {
            'cognitive_interface': ci,
            'queries': results,
            'status': status,
        }


# ---------------------------------------------------------------------------
# Unified Trainer
# ---------------------------------------------------------------------------

class UnifiedTrainer:
    """PHYSMOL unified training pipeline.

    Trains all modules in sequence:
      Phase 1: Physics Learning (LGNN)
      Phase 2: Concept Formation (VSA Recipes)
      Phase 3: Language Alignment (InfoNCE)
      Phase 4: End-to-End Integration
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.config.setdefault('vsa_dim', 10000)
        self.config.setdefault('coord_dim', 2)
        self.config.setdefault('hidden_dim', 64)
        self.config.setdefault('num_layers', 3)
        self.config.setdefault('seed', 42)
        self.config.setdefault('phase1_epochs', 500)
        self.config.setdefault('phase1_batch_size', 32)
        self.config.setdefault('phase2_scenarios', 50)
        self.config.setdefault('phase3_epochs', 100)
        self.config.setdefault('phase3_lr', 0.001)
        self.config.setdefault('save_path', './checkpoints')

        self.results = {}

    def run_phase1(self, device: str = 'auto') -> Dict:
        """Run Phase 1: Physics Learning."""
        phase1 = Phase1Physics(
            coord_dim=self.config['coord_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            seed=self.config['seed'],
        )

        save_path = os.path.join(self.config['save_path'], 'phase1')
        self.results['phase1'] = phase1.run(
            device=device,
            epochs=self.config['phase1_epochs'],
            batch_size=self.config['phase1_batch_size'],
            save_path=save_path,
        )
        return self.results['phase1']

    def run_phase2(self) -> Dict:
        """Run Phase 2: Concept Formation."""
        phase2 = Phase2Concepts(
            vsa_dim=self.config['vsa_dim'],
            seed=self.config['seed'],
        )

        save_path = os.path.join(self.config['save_path'], 'phase2')
        self.results['phase2'] = phase2.run(
            n_scenarios=self.config['phase2_scenarios'],
            save_path=save_path,
        )
        return self.results['phase2']

    def run_phase3(self) -> Dict:
        """Run Phase 3: Language Alignment."""
        if 'phase2' not in self.results:
            raise RuntimeError("Must run Phase 2 before Phase 3")

        phase3 = Phase3Language(
            vsa_dim=self.config['vsa_dim'],
            seed=self.config['seed'],
        )

        save_path = os.path.join(self.config['save_path'], 'phase3')
        self.results['phase3'] = phase3.run(
            store=self.results['phase2']['store'],
            scenarios=self.results['phase2']['scenarios'],
            epochs=self.config['phase3_epochs'],
            lr=self.config['phase3_lr'],
            save_path=save_path,
        )
        return self.results['phase3']

    def run_phase4(self) -> Dict:
        """Run Phase 4: End-to-End Integration."""
        if 'phase2' not in self.results:
            raise RuntimeError("Must run Phase 2 before Phase 4")

        lgnn = None
        if 'phase1' in self.results and 'trainer' in self.results['phase1']:
            lgnn = self.results['phase1']['trainer'].lgnn

        phase4 = Phase4Integration(
            vsa_dim=self.config['vsa_dim'],
            seed=self.config['seed'],
        )

        save_path = os.path.join(self.config['save_path'], 'phase4')
        self.results['phase4'] = phase4.run(
            lgnn=lgnn,
            store=self.results['phase2']['store'],
            text_encoder=self.results.get('phase3', {}).get('text_encoder'),
            scenarios=self.results['phase2']['scenarios'],
            save_path=save_path,
        )
        return self.results['phase4']

    def run_full(self, device: str = 'auto') -> Dict:
        """Run all 4 phases sequentially."""
        print("=" * 60)
        print("PHYSMOL Unified Training Pipeline")
        print(f"  VSA dim: {self.config['vsa_dim']}")
        print(f"  LGNN hidden: {self.config['hidden_dim']}")
        print(f"  Device: {device}")
        print("=" * 60)

        start_time = time.time()

        # Phase 1
        self.run_phase1(device)

        # Phase 2
        self.run_phase2()

        # Phase 3
        self.run_phase3()

        # Phase 4
        self.run_phase4()

        total_time = time.time() - start_time

        # Summary
        print("=" * 60)
        print("Training Complete!")
        print(f"  Total time: {total_time:.1f}s")
        if 'phase1' in self.results:
            print(f"  Phase 1 (Physics): {self.results['phase1'].get('total_time', 0):.1f}s")
        if 'phase2' in self.results:
            print(f"  Phase 2 (Concepts): {self.results['phase2']['num_recipes']} recipes")
        if 'phase3' in self.results:
            print(f"  Phase 3 (Language): loss={self.results['phase3']['losses'][-1]:.4f}")
        if 'phase4' in self.results:
            print(f"  Phase 4 (Integration): {len(self.results['phase4']['queries'])} queries validated")
        print(f"  Saved to: {self.config['save_path']}")
        print("=" * 60)

        return self.results

    def save(self, path: Optional[str] = None):
        """Save all trained components."""
        path = path or self.config['save_path']
        os.makedirs(path, exist_ok=True)

        # Save config
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

        # Save LGNN model
        if 'phase1' in self.results and 'trainer' in self.results['phase1']:
            self.results['phase1']['trainer'].lgnn.save(
                os.path.join(path, 'lgnn_trained.pt'))

        # Save recipes
        if 'phase2' in self.results:
            store = self.results['phase2']['store']
            recipes_data = {
                obj_id: store.get_recipe(obj_id)
                for obj_id in store.list_objects()
            }
            with open(os.path.join(path, 'recipes.json'), 'w') as f:
                json.dump(recipes_data, f, indent=2)

        # Save word vectors
        if 'phase3' in self.results:
            text_enc = self.results['phase3']['text_encoder']
            word_vecs = {}
            for word in text_enc.lexicon.words():
                word_vecs[word] = text_enc.lexicon.get_vector(word).tolist()
            with open(os.path.join(path, 'word_vectors.json'), 'w') as f:
                json.dump({'vsa_dim': self.config['vsa_dim'],
                           'vocabulary': word_vecs}, f)

        print(f"All components saved to {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='PHYSMOL Unified Training Pipeline')
    parser.add_argument('--phase', type=int, default=0,
                        help='Run specific phase (1-4), 0=all')
    parser.add_argument('--device', default='auto', help='cpu/cuda/rocm/auto')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override phase1 epochs')
    parser.add_argument('--phase3-epochs', type=int, default=None,
                        help='Override phase3 epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override phase1 batch size')
    parser.add_argument('--vsa-dim', type=int, default=10000,
                        help='VSA vector dimension')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='LGNN hidden dimension')
    parser.add_argument('--scenarios', type=int, default=None,
                        help='Number of object scenarios for Phase 2')
    parser.add_argument('--save-path', default='./checkpoints',
                        help='Save path for checkpoints')
    parser.add_argument('--checkpoint', default=None,
                        help='Load checkpoint for Phase 4')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = {
        'vsa_dim': args.vsa_dim,
        'hidden_dim': args.hidden_dim,
        'seed': args.seed,
        'save_path': args.save_path,
    }

    if args.epochs:
        config['phase1_epochs'] = args.epochs
    if args.phase3_epochs:
        config['phase3_epochs'] = args.phase3_epochs
    if args.batch_size:
        config['phase1_batch_size'] = args.batch_size
    if args.scenarios:
        config['phase2_scenarios'] = args.scenarios

    trainer = UnifiedTrainer(config)

    if args.phase == 0:
        trainer.run_full(device=args.device)
        trainer.save()
    elif args.phase == 1:
        trainer.run_phase1(device=args.device)
        trainer.save()
    elif args.phase == 2:
        trainer.run_phase2()
        trainer.save()
    elif args.phase == 3:
        trainer.run_phase2()  # Need Phase 2 first
        trainer.run_phase3()
        trainer.save()
    elif args.phase == 4:
        if args.checkpoint:
            # Load checkpoint and run Phase 4
            trainer.run_phase2()  # Need Phase 2 for recipes
            trainer.run_phase4()
        else:
            print("Phase 4 requires --checkpoint or run all phases with --phase 0")
    else:
        print(f"Unknown phase: {args.phase}")


if __name__ == '__main__':
    main()
