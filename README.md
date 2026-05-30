# PHYSMOL

**Physical Isomorphism and Symbolic Binding for Embodied Concept Learning**

A neuro-symbolic cognitive architecture that enables machines to build grounded physical understanding from scratch through active exploration.

PHYSMOL emulates how infants learn about the world: not by consuming massive labeled datasets, but by actively exploring physical environments, forming discrete concepts from continuous experience, and grounding language in physical interaction. It combines three computational substrates -- **Vector Symbolic Architectures (VSA)**, **Lagrangian Graph Neural Networks (LGNN)**, and **Spiking Neural Networks (SNN)** -- into a unified framework with a language cognitive layer on top.

---

## Architecture

```
                           ┌─────────────────────────────────┐
                           │      Language Cognitive Layer    │
                           │  TextToVSA ─ SemanticParser     │
                           │  ReasoningEngine ─ VSAGenerator │
                           │  TheoryOfMind ─ CodeReasoning   │
                           │  KnowledgeAcquisition           │
                           └──────────┬──────────┬───────────┘
                                      │          │
                          InfoNCE     │          │   Causal Graph
                       Alignment Hub  │          │   (SNN + STDP)
                                      │          │
           ┌──────────────────────────┼──────────┼──────────────────────┐
           │                          ▼          ▼                      │
           │  ┌────────────────────────────────────────────────────┐    │
           │  │           VSA Recipe Store (4096-dim bipolar)      │    │
           │  │  Attribute Primitives ─ Object Recipes ─ Resonance│    │
           │  └─────────────────────┬──────────────────────────────┘    │
           │                        │                                   │
           │         ┌──────────────┼──────────────┐                   │
           │         ▼              ▼              ▼                   │
           │  ┌────────────┐ ┌────────────┐ ┌───────────────────┐     │
           │  │ Perception │ │ World Model│ │ Intrinsic         │     │
           │  │ 5-modality │ │ L1/L2/L3   │ │ Motivation        │     │
           │  └────────────┘ └────────────┘ └───────────────────┘     │
           │                                                          │
           │              Cognitive Core (numpy / C)                   │
           └──────────────────────────────────────────────────────────┘
                                      │
                                      ▼
           ┌──────────────────────────────────────────────────────────┐
           │              Physics Engine Layer                        │
           │                                                          │
           │  ┌──────────────────────┐  ┌──────────────────────────┐ │
           │  │  LNN (C core)        │  │  LGNN (PyTorch)          │ │
           │  │  Euler-Lagrange MLP  │  │  Graph Energy Decomp.    │ │
           │  │  6-dim coordinates   │  │  Cross-topology N-body   │ │
           │  └──────────────────────┘  └──────────────────────────┘ │
           │                                                          │
           │  ┌──────────────────────┐  ┌──────────────────────────┐ │
           │  │  SNN (C core)        │  │  MuJoCo Simulation       │ │
           │  │  LIF + STDP          │  │  (optional)              │ │
           │  │  Three-factor learn  │  │  Physics experiments     │ │
           │  └──────────────────────┘  └──────────────────────────┘ │
           └──────────────────────────────────────────────────────────┘
```

---

## Key Features

### Physics Engine

**LGNN (Lagrangian Graph Neural Network)** -- PyTorch-based graph network that models physical systems as graphs where nodes are particles and edges are interactions (springs, gravity, constraints). The Lagrangian decomposes into per-node kinetic and per-edge potential energy, and accelerations are derived via the Euler-Lagrange equation. Energy conservation is structural, not learned. Supports cross-topology generalization: train on 2-body systems, apply to N-body.

**LNN (Lagrangian Network)** -- C-accelerated MLP that maps generalized coordinates to a scalar Lagrangian. Computes accelerations via autodiff of the Euler-Lagrange equation. Includes SIMD-optimized forward pass with numpy fallback.

### VSA Concept Memory

Objects are stored as **recipes** (lists of attribute IDs), not raw vectors. The `AttributePrimitivePool` holds global bipolar vectors for concepts like "sphere", "red", "elastic". The `RecipeStore` maps each object to a combination of attribute IDs. Concept vectors are synthesized on-demand via bundle + bind operations. Resonance search retrieves the closest matching recipe for any query vector.

This is NOT a database. Raw state (positions, velocities, color values) is never stored in VSA. The multimodal encoder extracts discrete attribute categories, and VSA stores only the category IDs.

### Spiking Neural Network

Leaky Integrate-and-Fire neurons with bit-compressed spike states (4096 neurons = 512 bytes). Three-factor STDP learning with eligibility traces for causal credit assignment. Event-driven causal graph construction with pruning and reinforcement.

### Language Cognitive Layer

Full pipeline from natural language to grounded physical and code reasoning:

- **TextToVSA / EnhancedTextEncoder**: tokenization, word vectors, positional encoding, optional sentence-transformer backend, fastText/Word2Vec/GloVe pre-trained vectors, jieba Chinese tokenization
- **SemanticParser**: intent classification (predict/counterfactual/explain/plan/chat/code), attribute extraction, object matching via resonance
- **ReasoningEngine**: physics prediction, counterfactual reasoning, concept explanation, action planning, **code reasoning** (explain algorithms, compare data structures, suggest solutions)
- **VSALanguageGenerator**: VSA-driven language generation (replaces template Responder), code generation with 20+ patterns (quicksort, BFS, linked list, LRU cache, etc.)
- **TheoryOfMind**: belief-desire-intention modeling for other agents
- **AbstractReasoner**: multi-hop inference over abstract concepts (fairness, justice, democracy)
- **AbstractTaskReasoner**: domain-specific reasoning for math proofs, legal cases, moral judgment
- **KnowledgeAcquisition**: automatic concept learning from text and interaction, vocabulary expansion

### Unified VSA Concept Space

Physical and code concepts share the same VSA vector space:

```
Physical attributes          Code attributes
─────────────────           ─────────────────
shape (sphere, cube...)     algorithm (sort, search, traverse...)
color (red, blue...)        data_structure (array, list, stack...)
material (metal, wood...)   operation (create, read, update...)
elasticity (rigid...)       control_flow (loop, conditional...)
mass (light, heavy...)      complexity (constant, logarithmic...)
```

This enables cross-domain reasoning: the system understands that "sort" and "gravity" are both concepts that can be explained, compared, and composed.

### Hierarchical World Model

Three-level decision hierarchy inspired by cognitive science:

- **L1 Predictor**: fast single-step linear prediction (millisecond timescale)
- **L2 Simulator**: multi-step forward simulation via mental experiments (seconds)
- **L3 Evolver**: structural knowledge updates when L2 repeatedly fails (minutes-hours)

A `CuriositySignal` quantifies prediction uncertainty to trigger higher decision levels.

### Additional Systems

- **Intrinsic Motivation**: novelty + uncertainty + learning progress scoring for action selection
- **Long-Term Memory**: episodic, factual, and experience memory with VSA-based retrieval and consolidation
- **Cross-Domain Transfer**: schema-based transfer between domains (e.g., blocks to chess)
- **Multimodal Perception**: vision, audio, tactile, olfactory, and proprioceptive encoders with MuJoCo integration
- **Knowledge Acquisition**: automatic concept learning from text, interactive teaching, vocabulary expansion

---

## Project Structure

```
PHYSMOL/
├── config/
│   └── default.yaml                # All configuration parameters
│
├── src/
│   ├── core/                       # C core libraries (SIMD-optimized)
│   │   ├── vsa.c / vsa.h          # VSA: bind, bundle, unbind, FPE, codebook
│   │   ├── snn.c / snn.h          # SNN: LIF neurons, STDP, three-factor learning
│   │   ├── lnn.c / lnn.h          # LNN: MLP Lagrangian, Euler-Lagrange acceleration
│   │   ├── causal.c / causal.h    # Causal graph: adjacency list, propagate, prune
│   │   └── memory.c / memory.h    # Tiered memory: L1 pool, L2 hashmap, L3 file-backed
│   │
│   ├── bindings/                   # pybind11 Python bindings (numpy zero-copy)
│   │   ├── vsa_py.cpp
│   │   ├── snn_py.cpp
│   │   └── lnn_py.cpp
│   │
│   ├── hip/                        # AMD GPU (ROCm/HIP) kernel templates
│   │   ├── lnn_hip.hip
│   │   └── lnn_hip.h
│   │
│   └── python/physmol/             # Python package
│       ├── __init__.py             # Public API exports
│       ├── vsa.py                  # VSAVector, VectorSymbolicArchitecture, Codebook, FHRRSpace
│       ├── vsa_store.py            # AttributePrimitivePool, RecipeStore, ConceptSynthesizer
│       ├── vsa_concepts.py         # PrimitiveCodebook, FieldEncoder, ObjectConcept
│       ├── lnn.py                  # LagrangianNetwork (C or numpy fallback)
│       ├── lgnn.py                 # LagrangianGraphNetwork, PhysicsGraph, analytical systems
│       ├── lgnn_train.py           # LGNN training pipeline + CLI
│       ├── snn.py                  # LIFNeuronParams, SpikeTrain, SpikingNetwork, CausalGraph
│       ├── alignment.py            # AlignmentHub (InfoNCE cross-modal alignment)
│       ├── world_model.py          # HierarchicalWorldModel (L1/L2/L3)
│       ├── perception.py           # MultiModalPerception (5 modality encoders)
│       ├── sim_env.py              # PhysicsExperiment (MuJoCo wrapper)
│       ├── train.py                # SerialTrainer (two-phase: physical + language)
│       ├── unified_train.py        # UnifiedTrainer (4-phase pipeline) + CLI
│       ├── motivation.py           # IntrinsicMotivationSystem
│       ├── long_term_memory.py     # LongTermMemory with episode/fact/experience types
│       ├── transfer.py             # CrossDomainTransferEngine
│       ├── progress.py             # ProgressLogger
│       ├── progress_server.py      # HTTP dashboard for training progress + CLI
│       ├── training_data.py        # Dataset adapters (ModelScope, local JSONL/text)
│       ├── abstract_training.py    # AbstractCognitionTrainer (proof, legal, moral)
│       ├── abstract_train.py       # CLI for abstract cognition training
│       │
│       ├── knowledge_acquisition.py # Automatic concept learning from text/interaction
│       │
│       └── language/               # Language cognitive layer
│           ├── text_encoder.py     # TextToVSA, WordLexicon (Chinese support)
│           ├── enhanced_encoder.py # Enhanced encoder with fastText/sentence-transformers
│           ├── vsa_generator.py    # VSA-driven language generation + code patterns
│           ├── semantic_parser.py  # Intent classification + attribute extraction
│           ├── reasoning.py        # Causal + code reasoning engine
│           ├── responder.py        # Template-based NLG (legacy)
│           ├── cognitive.py        # CognitiveInterface (unified API)
│           ├── conversation.py     # DialogueState, DialogueTurn
│           ├── theory_of_mind.py   # TheoryOfMindModel, AgentMind
│           ├── abstract_reasoning.py  # AbstractConceptReasoner
│           └── abstract_tasks.py   # AbstractTaskReasoner (math/legal/moral)
│
├── tests/
│   ├── test_vsa.c / test_vsa.py    # VSA tests (C + Python)
│   ├── test_snn.c / test_snn.py    # SNN tests (C + Python)
│   ├── test_lnn.py                 # LNN tests
│   ├── test_lgnn.py                # LGNN tests (spring, gravity, cross-topology)
│   ├── test_language.py            # Language layer tests
│   └── test_cognitive_extensions.py # Abstract reasoning, motivation, ToM, memory, transfer
│
├── scripts/
│   ├── modelscope_abstract_training.sh
│   └── cloud_train.sh            # Cloud server one-command training script
│
├── docs/
│   └── MODELSCOPE_TRAINING.md      # ModelScope cloud deployment guide
│
├── PHYSMOL.md                      # Full theoretical paper (bilingual EN/ZH)
├── CHANGELOG.md                    # Development log
├── AI_evaluation.md                # AI evaluation document
├── CMakeLists.txt                  # CMake build (alternative)
├── Makefile                        # Convenience build targets
└── setup.py                        # Python package build with pybind11
```

---

## Quick Start

### Prerequisites

- Python >= 3.8
- numpy >= 1.20
- pybind11 >= 2.10

### Install

```bash
# Clone the repository
git clone https://github.com/12wdsj/PHYSMOL.git
cd PHYSMOL

# Install the package (pure Python, no C extensions needed for most features)
pip install -e .

# Optional dependencies
pip install torch>=2.0        # Required for LGNN physics engine
pip install mujoco>=2.3       # Required for physics simulation
```

### Build C Extensions (optional, for performance)

The C extensions provide SIMD-accelerated VSA and SNN operations. The LNN falls back to numpy if C extensions are not built.

```bash
# Using Makefile
make build

# Or manually
pip install pybind11
python setup.py build_ext --inplace

# Verify the build
make smoke
```

### Verify Installation

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_language.py -v     # Language layer (no C ext needed)
python -m pytest tests/test_lgnn.py -v         # LGNN (requires PyTorch)
python -m pytest tests/test_cognitive_extensions.py -v  # Cognitive extensions
```

---

## Usage

### Language Cognitive Interface

The `CognitiveInterface` is the main entry point for natural language interaction:

```python
from physmol import CognitiveInterface

# Initialize
ci = CognitiveInterface(vsa_dim=4096)

# Register objects discovered through physical exploration
ci.register_object('red_ball', [
    'shape_sphere', 'color_red', 'material_rubber', 'elasticity_elastic'
])
ci.register_object('blue_cube', [
    'shape_cube', 'color_blue', 'material_metal', 'elasticity_rigid'
])

# Physics prediction
ci.query("What happens if I drop the red ball?")
# -> "Objects accelerate downward at 9.8 m/s^2. The red ball is elastic..."

# Concept explanation
ci.query("Explain elasticity")
# -> "Elasticity is the ability of a material to return to its original shape..."

# Action planning
ci.query("Push the cube to the top")
# -> "Understood. I will push the cube. Plan: 1. Identify the cube..."

# Counterfactual reasoning
ci.query("What if the ball was heavier?")
# -> "More inertia, harder to accelerate, same fall speed, greater impact force"
```

### Code Generation

PHYSMOL can generate code from natural language descriptions, using VSA pattern matching:

```python
# Code generation
ci.query("Write a function called quicksort that sorts a list")
# -> def quicksort(arr):
#        if len(arr) <= 1: return arr
#        pivot = arr[len(arr) // 2]
#        ...

ci.query("Implement a binary search")
# -> def binary_search(arr, target):
#        lo, hi = 0, len(arr) - 1
#        ...

ci.query("Write a class called Stack with push and pop methods")
# -> class Stack:
#        def __init__(self): self._items = []
#        def push(self, item): ...
#        def pop(self): ...

# Chinese code generation
ci.query("写一个函数实现二分查找")
# -> def binary_search(arr, target): ...

# Code explanation
ci.query("Explain quicksort")
# -> **quicksort**
#    Definition: A divide-and-conquer sorting algorithm...
#    Complexity: best O(n log n), worst O(n^2)
#    Use cases: general purpose sorting, in-place sorting
```

### Knowledge Acquisition

PHYSMOL can learn new concepts from text and interaction:

```python
# Teach a concept explicitly
ci.teach_concept(
    term="recursion",
    category="algorithm",
    definition="A technique where a function calls itself",
    examples=["factorial", "fibonacci", "tree traversal"],
    related=["divide and conquer", "stack"]
)

# Learn concepts from text automatically
text = """
A neural network is a machine learning model inspired by biological neurons.
Deep learning refers to neural networks with many layers.
Backpropagation is the algorithm used to train neural networks.
"""
learned = ci.learn_from_text(text)
# -> Learns: neural network, deep learning, backpropagation, ...

# Query learned concepts
info = ci.get_concept_info("recursion")
# -> LearnedConcept(term="recursion", category="algorithm", ...)

# List all learned concepts
concepts = ci.list_learned_concepts("algorithm")
```

### Enhanced Language Encoder

For production use with large vocabulary:

```python
from physmol.language.enhanced_encoder import EnhancedTextEncoder

encoder = EnhancedTextEncoder(vsa_dim=4096)

# Load pre-trained word vectors (covers ~100,000 words per language)
encoder.load_pretrained_vectors("./vectors/cc.zh.300.vec")  # Chinese
encoder.load_pretrained_vectors("./vectors/cc.en.300.vec")  # English

# Load sentence-transformers for context-aware encoding
encoder.init_sentence_transformer("paraphrase-multilingual-MiniLM-L12-v2")

# Encode any text
vec = encoder.encode("快速排序算法的时间复杂度是 O(n log n)")
```

### VSA Recipe Store

```python
from physmol import AttributePrimitivePool, RecipeStore, ConceptSynthesizer

# Create the attribute primitive pool (global shared alphabet)
pool = AttributePrimitivePool(vsa_dim=4096, seed=42)
store = RecipeStore(pool)
synth = ConceptSynthesizer(store)

# Register objects as recipes (attribute ID lists, NOT raw data)
store.register_recipe('ball_1', ['shape_sphere', 'color_red', 'material_rubber'])
store.register_recipe('cube_1', ['shape_cube', 'color_blue', 'material_metal'])

# Synthesize VSA vector on-demand
vec = store.synthesize('ball_1')

# Resonance search: find most similar recipe
matches = store.resonate(vec, top_k=3)

# Decompose: recover attributes from a vector
attrs = store.decompose(vec)
# -> {'shape': ('sphere', 0.51), 'color': ('red', 0.50), 'material': ('rubber', 0.49)}

# Cross-object similarity
sim = synth.concept_similarity('ball_1', 'cube_1')
```

### LGNN Physics Engine

```python
from physmol.lgnn import LagrangianGraphNetwork, PhysicsGraph, SpringMassSystem
import numpy as np

# Create a 2-body spring system graph
graph = PhysicsGraph.make_chain(num_nodes=2, coord_dim=2)
lgnn = LagrangianGraphNetwork(coord_dim=2, hidden_dim=64)

# Simulate trajectory
q0 = np.array([[0.0, 0.0], [1.2, 0.0]], dtype=np.float32)
v0 = np.array([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32)
q_traj, v_traj = lgnn.simulate_trajectory(q0, v0, graph, dt=0.01, steps=100)

# Analytical baseline for comparison
system = SpringMassSystem(m1=1.0, m2=1.0, k=5.0, L0=1.0)
q_exact, v_exact = system.simulate(q0, v0, dt=0.01, steps=100)

# Train LGNN against analytical ground truth
from physmol.lgnn_train import LGNNTrainer
trainer = LGNNTrainer(coord_dim=2, hidden_dim=64, device='cpu')
results = trainer.train(epochs=500, batch_size=32)
```

### SNN and Causal Learning

```python
from physmol.snn import LIFNeuronParams, SpikingNetwork, CausalGraph, ThreeFactorLearner
import numpy as np

# Create a spiking network
params = LIFNeuronParams(v_threshold=1.0, v_reset=0.0, tau_mem=20.0, tau_ref=2.0, dt=1.0)
net = SpikingNetwork(num_pre=100, num_post=50, params=params)

# Run one timestep
input_current = np.random.randn(100).astype(np.float32)
voltages, spikes = net.step(input_current)

# STDP learning
net.stdp(a_plus=0.01, a_minus=0.012, tau_plus=20.0, tau_minus=20.0)

# Causal graph construction
graph = CausalGraph(max_nodes=256, capacity=1024)
graph.add_edge(pre=0, post=1, weight=0.5, credit=0.1)
graph.reinforce(active_nodes=np.array([0, 1]), reward=1.0)
activation = graph.propagate(source=0, steps=3)
```

### Hierarchical World Model

```python
from physmol import HierarchicalWorldModel
import numpy as np

model = HierarchicalWorldModel(state_dim=6)

# L1 fast prediction
state = np.random.randn(6).astype(np.float32)
action = np.random.randn(6).astype(np.float32) * 0.1
observed_next = state + action + np.random.randn(6).astype(np.float32) * 0.01

result = model.step(state, action, observed_next)
# -> {'level': 1, 'prediction_error': 0.0003, 'status': 'normal'}

# When prediction error is high, L2/L3 activate automatically
print(model.get_status())
```

### Theory of Mind

```python
from physmol.language import CognitiveInterface

ci = CognitiveInterface(vsa_dim=4096)

# Register agents
ci.register_agent('alice')

# Observe agent mental states
ci.observe_agent('alice', belief='ball_is_red', desire='find_ball', emotion='curious')

# Theory of mind queries
response = ci.chat("What does alice believe?")
# -> "Alice believes the ball is red."

response = ci.chat("Is alice happy?")
# -> "Alice appears to be curious based on observations."
```

### Long-Term Memory

```python
from physmol import LongTermMemory, TextToVSA

encoder = TextToVSA(vsa_dim=4096)
memory = LongTermMemory(encoder)

# Store different memory types
memory.add_episode(
    content="The red ball bounced off the wall",
    actors=["ball_1", "wall"],
    context="physics_experiment",
    outcome="ball_changed_direction"
)

memory.add_fact(
    subject="rubber", predicate="is_elastic", obj="True",
    evidence="observed_bounce", confidence=0.95
)

memory.add_experience(
    description="Pushed the cube",
    action="push_force_5n",
    observation="cube_moved_2m",
    reward=0.8,
    prediction_error=0.05
)

# Retrieve by query
results = memory.retrieve("What happened to the ball?", top_k=3)

# Consolidation: strengthen frequently accessed memories
memory.consolidate(min_repetitions=3)

# Persist to disk
memory.save_json("./checkpoints/long_term_memory.json")
```

---

## Training Pipeline

PHYSMOL provides both a 4-phase unified trainer and individual phase CLIs.

### Unified Training (4 phases)

```bash
# Full pipeline
python -m physmol.unified_train --device cuda --epochs 500 --save-path ./checkpoints

# Individual phases
python -m physmol.unified_train --phase 1 --device cuda --epochs 500    # Physics (LGNN)
python -m physmol.unified_train --phase 2 --scenarios 50                # Concept formation
python -m physmol.unified_train --phase 3 --device cuda --phase3-epochs 100  # Language alignment
python -m physmol.unified_train --phase 4 --checkpoint ./checkpoints    # Integration
```

| Phase | Name | Compute | Description |
|-------|------|---------|-------------|
| 1 | Physics Learning | GPU | Train LGNN on analytical ground truth (spring-mass, gravity) |
| 2 | Concept Formation | CPU | Extract physical attributes, register VSA recipes |
| 3 | Language Alignment | GPU | InfoNCE contrastive learning: text vectors <-> physical concept vectors |
| 4 | Integration | CPU | Wire LGNN into CognitiveInterface, end-to-end validation |

### LGNN Standalone Training

```bash
python -m physmol.lgnn_train --device cuda --epochs 1000 --batch-size 128
```

### Abstract Cognition Training

```bash
# Built-in examples (math proofs, legal reasoning, moral judgment)
python -m physmol.abstract_train --use-builtin

# From ModelScope dataset
python -m physmol.abstract_train --modelscope-dataset <dataset_id> --subset-name <subset>

# From local JSONL file
python -m physmol.abstract_train --local-jsonl ./data/examples.jsonl
```

### Training Progress Dashboard

```bash
python -m physmol.progress_server --port 7860
# Open http://localhost:7860 in browser
```

### Cloud Server Training

One-command training script for cloud deployment (ModelScope, AutoDL, etc.):

```bash
# Full training (auto-downloads word vectors + all phases)
DEVICE=cuda bash scripts/cloud_train.sh all

# Individual steps
bash scripts/cloud_train.sh setup      # Install dependencies
bash scripts/cloud_train.sh vectors    # Download pre-trained word vectors
bash scripts/cloud_train.sh phase1     # Physics learning only
bash scripts/cloud_train.sh language   # Language training only
bash scripts/cloud_train.sh abstract   # Abstract cognition only
```

Environment variables:
- `PROJECT_DIR`: Project directory (default: `/mnt/workspace/PHYSMOL`)
- `DEVICE`: Device (auto, cuda, rocm, cpu)
- `EPOCHS`: Training epochs (default: 500)

See [docs/MODELSCOPE_TRAINING.md](docs/MODELSCOPE_TRAINING.md) for ModelScope-specific instructions.

---

## Configuration

All parameters are in `config/default.yaml`. Key sections:

```yaml
vsa:
  dim: 4096                    # VSA vector dimension
  float_precision: float32

snn:
  num_neurons: 512
  v_threshold: 1.0             # LIF spike threshold
  tau_mem: 20.0                # Membrane time constant (ms)
  stdp:
    a_plus: 0.01               # LTP learning rate
    a_minus: 0.012             # LTD learning rate
  three_factor:
    tau_e: 100.0               # Eligibility trace decay (ms)
    eta: 0.001                 # Learning rate

lnn:
  coord_dim: 6                 # Generalized coordinates
  hidden_dim: 128

alignment:
  lang_dim: 300                # Word vector dimension
  temperature: 0.07            # InfoNCE temperature

world_model:
  l2_threshold: 0.1            # Prediction error to activate L2
  l3_threshold: 0.5            # Error to activate L3

motivation:
  novelty_weight: 0.35
  uncertainty_weight: 0.25
  progress_weight: 0.40

training:
  lr: 0.001
  batch_size: 32
  max_episodes: 10000

simulation:
  engine: mujoco
  timestep: 0.01
  parallel_envs: 4
```

See `config/default.yaml` for the complete reference with all options.

---

## API Reference

### Core Classes

| Module | Class | Description |
|--------|-------|-------------|
| `vsa.py` | `VectorSymbolicArchitecture` | VSA operations: bind, bundle, unbind, FPE, similarity |
| `vsa.py` | `Codebook` | Named vector storage with nearest-neighbor lookup |
| `vsa.py` | `FHRRSpace` | Frequency Holographic Reduced Representation for spatial encoding |
| `vsa_store.py` | `AttributePrimitivePool` | Global shared attribute vectors (shape, color, material, ...) |
| `vsa_store.py` | `RecipeStore` | Object recipe storage with resonance search and decomposition |
| `vsa_store.py` | `ConceptSynthesizer` | High-level concept composition and matching |
| `lgnn.py` | `LagrangianGraphNetwork` | Graph neural network with Lagrangian energy decomposition |
| `lgnn.py` | `PhysicsGraph` | Graph structure for physical systems (chain, fully-connected) |
| `lgnn.py` | `SpringMassSystem` | Analytical spring-mass baseline |
| `lgnn.py` | `GravitationalTwoBody` | Analytical two-body gravity baseline |
| `lnn.py` | `LagrangianNetwork` | C-accelerated Lagrangian MLP with Euler-Lagrange acceleration |
| `snn.py` | `SpikingNetwork` | LIF neural network with STDP learning |
| `snn.py` | `CausalGraph` | Causal relationship graph with propagation and pruning |
| `snn.py` | `ThreeFactorLearner` | Three-factor STDP with eligibility traces |
| `alignment.py` | `AlignmentHub` | InfoNCE cross-modal alignment (language <-> physics) |
| `world_model.py` | `HierarchicalWorldModel` | L1/L2/L3 decision hierarchy |
| `perception.py` | `MultiModalPerception` | 5-modality perception encoder |
| `motivation.py` | `IntrinsicMotivationSystem` | Novelty + uncertainty + progress scoring |
| `long_term_memory.py` | `LongTermMemory` | Episodic/factual/experience memory with consolidation |
| `transfer.py` | `CrossDomainTransferEngine` | Schema-based cross-domain transfer |

### Language Classes

| Module | Class | Description |
|--------|-------|-------------|
| `language/cognitive.py` | `CognitiveInterface` | Unified natural language API |
| `language/text_encoder.py` | `TextToVSA` | Text to VSA vector encoding |
| `language/text_encoder.py` | `WordLexicon` | Word vector vocabulary with Chinese support |
| `language/enhanced_encoder.py` | `EnhancedTextEncoder` | Large vocabulary encoder with fastText/sentence-transformers |
| `language/vsa_generator.py` | `VSALanguageGenerator` | VSA-driven language generation + code patterns |
| `language/vsa_generator.py` | `CodePattern` | Code generation pattern definition |
| `language/semantic_parser.py` | `SemanticParser` | Intent classification + attribute extraction |
| `language/reasoning.py` | `ReasoningEngine` | Physics + code reasoning, explanation, planning |
| `language/responder.py` | `Responder` | Template-based response generation (legacy) |
| `language/theory_of_mind.py` | `TheoryOfMindModel` | Belief-desire-intention modeling |
| `language/abstract_reasoning.py` | `AbstractConceptReasoner` | Multi-hop abstract concept inference |
| `language/abstract_tasks.py` | `AbstractTaskReasoner` | Math/legal/moral domain reasoning |
| `knowledge_acquisition.py` | `KnowledgeAcquisition` | Automatic concept learning from text/interaction |

### CLI Entry Points

| Command | Description |
|---------|-------------|
| `python -m physmol.unified_train` | 4-phase unified training pipeline |
| `python -m physmol.lgnn_train` | LGNN standalone training |
| `python -m physmol.abstract_train` | Abstract cognition training |
| `python -m physmol.progress_server` | Training progress HTTP dashboard |

---

## Build System

Three build options are available:

**setup.py** (primary):
```bash
python setup.py build_ext --inplace
```

**CMake** (alternative):
```bash
mkdir build && cd build
cmake .. && make
```

**Makefile** (convenience):
```bash
make build      # Build C extensions
make dev        # Development install
make test       # Run pytest
make test-c     # Run C unit tests
make smoke      # Quick smoke test
make clean      # Remove build artifacts
```

---

## Dependencies

### Required

| Package | Version | Notes |
|---------|---------|-------|
| Python | >= 3.8 | |
| numpy | >= 1.20 | Core computation |
| pybind11 | >= 2.10 | C extension bindings |

### Optional

| Package | Version | Enables |
|---------|---------|---------|
| torch | >= 2.0 | LGNN physics engine, unified training |
| mujoco | >= 2.3 | Physics simulation environment |
| modelscope | >= 1.9 | ModelScope dataset loading |
| sentence-transformers | any | Higher quality text encoding (multilingual) |
| jieba | any | Chinese tokenization for enhanced encoder |
| gensim | any | Word2Vec binary format loading |
| pytest | any | Running tests |

### Build Tools

| Tool | Platform | Notes |
|------|----------|-------|
| GCC (AVX2+FMA) | Linux/macOS | SIMD optimization for C extensions |
| MSVC | Windows | C extension compilation |
| CMake | >= 3.16 | Alternative build system |
| HIP/ROCm | AMD | GPU kernel compilation |

---

## Documentation

| Document | Description |
|----------|-------------|
| [PHYSMOL.md](PHYSMOL.md) | Full theoretical paper (bilingual EN/ZH), 50+ pages of mathematical proofs |
| [CHANGELOG.md](CHANGELOG.md) | Development log with architecture decisions |
| [CHANGES.md](CHANGES.md) | Detailed change log for each update |
| [AI_evaluation.md](AI_evaluation.md) | AI evaluation document |
| [docs/MODELSCOPE_TRAINING.md](docs/MODELSCOPE_TRAINING.md) | ModelScope cloud deployment guide |

---

## Development Principles

- **Physics first**: conservation laws and causal structure are hardcoded in network architecture, not learned from data
- **Recipe, not database**: VSA stores relationship patterns (attribute IDs), not raw states
- **On-demand synthesis**: concept vectors are computed from recipes when needed, not stored permanently
- **Cross-modal grounding**: language is anchored to physical experience through contrastive alignment
- **Modular and testable**: every component has isolated tests and clear interfaces
- **Graceful degradation**: C extensions are optional; LGNN requires PyTorch but everything else works with numpy

---

## References

- [1] Hwang et al., "Learning the Dynamics of Particle-based Systems with Lagrangian Graph Neural Networks", 2023
- [2] Cranmer et al., "Lagrangian Neural Networks", 2020
- [3] Frady et al., "Vector Symbolic Architectures as a Computing Framework for Emerging Hardware", 2023
- [4] Kanerva, "Hyperdimensional Computing", 2009
- [5] Maass, "Networks of Spiking Neurons: The Third Generation of Neural Network Models", 1997

---

## License

This project is open for academic and research purposes.
