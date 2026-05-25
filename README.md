# PHYSMOL

**Physical Isomorphism and Symbolic Binding for Embodied Concept Learning**

A neuro-symbolic cognitive architecture that enables machines to build grounded physical understanding from scratch through active exploration, combining Vector Symbolic Architectures (VSA), Lagrangian Graph Neural Networks (LGNN), and Spiking Neural Networks (SNN).

## Overview

PHYSMOL emulates how infants learn about the world: not by consuming massive labeled datasets, but by actively exploring physical environments, forming discrete concepts from continuous experience, and grounding language in physical interaction.

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHYSMOL Architecture                         │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   │
│  │   Language    │   │   Physical    │   │   Alignment      │   │
│  │   Cognitive   │◄─►│   Exploration │◄─►│   Hub (InfoNCE)  │   │
│  │   Layer       │   │   (LGNN+SNN)  │   │                  │   │
│  └──────┬───────┘   └──────┬───────┘   └──────────────────┘   │
│         │                   │                                   │
│         ▼                   ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              VSA Recipe Store (10,000-dim)                │  │
│  │   Attribute Primitives + Object Recipes + Resonance      │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                   │                                   │
│         ▼                   ▼                                   │
│  ┌──────────────┐   ┌──────────────┐                          │
│  │   Causal      │   │  Hierarchical │                          │
│  │   Graph       │   │  World Model  │                          │
│  │   (SNN+STDP)  │   │  (L1/L2/L3)  │                          │
│  └──────────────┘   └──────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### Physics Engine (LGNN)
- Lagrangian Graph Neural Network encodes physics as graph structure
- Nodes = particles/bodies, Edges = interactions (springs, gravity, constraints)
- Euler-Lagrange equation enforces energy conservation structurally
- Cross-topology generalization: train on 2-body, apply to N-body

### VSA Concept Memory
- 10,000-dimensional bipolar vectors with quasi-orthogonal capacity
- Recipe-based storage: objects = lists of attribute IDs, not raw data
- On-demand vector synthesis from attribute primitives
- Resonance search for concept retrieval and matching

### Language Cognitive Layer
- TextToVSA encoder: natural language to VSA vectors
- Semantic parser: intent classification + attribute extraction
- Reasoning engine: physics prediction, counterfactuals, concept explanation
- Template-based response generation

### Spiking Neural Network
- Leaky Integrate-and-Fire neurons with STDP learning
- Three-factor learning with eligibility traces for causal credit assignment
- Event-driven causal graph construction

## Project Structure

```
PHYSMOL/
├── PHYSMOL.md                  # Full theoretical paper
├── CMakeLists.txt              # Build system for C extensions
├── Makefile                    # Convenience build targets
├── setup.py                    # Python package setup
│
├── src/
│   ├── core/                   # C core libraries
│   │   ├── vsa.c / vsa.h       # VSA operations (bind, bundle, FPE)
│   │   ├── lnn.c / lnn.h       # Lagrangian Neural Network
│   │   ├── snn.c / snn.h       # Spiking Neural Network (LIF + STDP)
│   │   ├── causal.c / causal.h # Causal graph
│   │   └── memory.c / memory.h # Tiered memory management
│   │
│   ├── bindings/               # pybind11 Python bindings
│   │   ├── vsa_py.cpp
│   │   ├── lnn_py.cpp
│   │   └── snn_py.cpp
│   │
│   ├── hip/                    # AMD GPU (ROCm/HIP) kernels
│   │   ├── lnn_hip.hip
│   │   └── lnn_hip.h
│   │
│   └── python/physmol/         # Python package
│       ├── __init__.py
│       ├── vsa.py              # VSA Python wrapper (C extension)
│       ├── vsa_store.py        # VSA recipe store (correct philosophy)
│       ├── vsa_concepts.py     # Primitive codebook + field encoder
│       ├── lnn.py              # LNN Python wrapper
│       ├── lgnn.py             # Lagrangian Graph Neural Network (PyTorch)
│       ├── lgnn_train.py       # LGNN training pipeline
│       ├── snn.py              # SNN Python wrapper
│       ├── perception.py       # Multimodal perception encoders
│       ├── alignment.py        # InfoNCE cross-modal alignment
│       ├── world_model.py      # Hierarchical world model (L1/L2/L3)
│       ├── sim_env.py          # MuJoCo simulation environment
│       ├── train.py            # Serial training loop
│       └── language/           # Language cognitive layer
│           ├── text_encoder.py     # TextToVSA encoder
│           ├── semantic_parser.py  # Intent classification + matching
│           ├── reasoning.py        # Causal reasoning engine
│           ├── responder.py        # Response generation
│           └── cognitive.py        # Unified cognitive interface
│
├── tests/
│   ├── test_lnn.py             # LNN tests
│   ├── test_lgnn.py            # LGNN tests (spring, gravity, cross-topology)
│   ├── test_vsa.py / test_vsa.c    # VSA tests
│   ├── test_snn.py / test_snn.c    # SNN tests
│   └── test_language.py        # Language layer tests
│
└── config/
    └── default.yaml            # Default configuration
```

## Quick Start

### Prerequisites

```bash
pip install numpy
pip install torch                          # Required for LGNN
pip install mujoco                         # Optional: physics simulation
```

### Build C Extensions (optional, for performance)

```bash
# Using Makefile
make build

# Or manually
pip install pybind11
python setup.py build_ext --inplace
```

### Basic Usage

```python
from physmol import CognitiveInterface

# Initialize the cognitive system
ci = CognitiveInterface(vsa_dim=10000)

# Register objects discovered through physical exploration
ci.register_object('red_ball', [
    'shape_sphere', 'color_red', 'material_rubber', 'elasticity_elastic'
])
ci.register_object('blue_cube', [
    'shape_cube', 'color_blue', 'material_metal', 'elasticity_rigid'
])

# Ask physics questions
ci.query("What happens if I drop the red ball?")
# -> "Objects accelerate downward at 9.8 m/s^2..."

# Explain concepts
ci.query("Explain elasticity")
# -> "Elasticity is the ability of a material to return to its original shape..."

# Execute commands
ci.query("Push the cube to the top")
# -> "Understood. I will push the cube. Plan: 1. Identify the cube..."

# Counterfactual reasoning
ci.query("What if the ball was heavier?")
# -> "More inertia, harder to accelerate, same fall speed, greater impact force"
```

### LGNN Physics Training

```python
from physmol.lgnn import LagrangianGraphNetwork, PhysicsGraph
import numpy as np

# Create a 2-body spring system graph
graph = PhysicsGraph.make_chain(2, coord_dim=2)
lgnn = LagrangianGraphNetwork(coord_dim=2, hidden_dim=64)

# Simulate trajectory
q0 = np.array([[0.0, 0.0], [1.2, 0.0]], dtype=np.float32)
v0 = np.array([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32)
q_traj, v_traj = lgnn.simulate_trajectory(q0, v0, graph, dt=0.01, steps=100)

# Train against analytical ground truth
python -m physmol.lgnn_train --device cuda --epochs 1000 --batch-size 128
```

### VSA Recipe Store

```python
from physmol import AttributePrimitivePool, RecipeStore

# Create the attribute primitive pool
pool = AttributePrimitivePool(vsa_dim=10000)
store = RecipeStore(pool)

# Register objects as recipes (attribute ID lists, NOT raw data)
store.register_recipe('ball_1', ['shape_sphere', 'color_red', 'material_rubber'])

# Synthesize VSA vector on-demand
vec = store.synthesize('ball_1')

# Resonance search: find most similar recipe
matches = store.resonate(query_vec, top_k=5)

# Decompose: recover attributes from a vector
decomp = store.decompose(query_vec)
# -> {'shape': ('sphere', 0.51), 'color': ('red', 0.50), ...}
```

## Architecture Details

### VSA: Relationship Pattern Memory

PHYSMOL's VSA is NOT a database. It stores:
1. **Attribute primitives**: global shared vectors for concepts like "sphere", "red", "elastic"
2. **Object recipes**: `{object_id: [attr_id1, attr_id2, ...]}` — combinations of attribute IDs

Raw state (positions, velocities, color values) is NEVER stored in VSA. Instead, the multimodal encoder extracts discrete attribute categories, and VSA stores only the category IDs.

### LGNN: Physics-Constrained Dynamics

The Lagrangian decomposes as:
```
L = SUM_i T_i(q_dot_i) - SUM_{(i,j) in E} V_ij(q_i, q_j)
```

Accelerations are derived via the Euler-Lagrange equation:
```
q_ddot = (d^2L/dq_dot^2)^{-1} [dL/dq - (d^2L/dq dq_dot) q_dot]
```

This ensures energy conservation is structural, not learned.

### Language: From Text to Physics

```
User text -> TextToVSA -> SemanticParser -> ReasoningEngine -> Responder -> Response
                |              |                  |
          word vectors    intent classify    causal graph
          + position      + attr extract     + physics rules
          encoding        + resonance        + LGNN sim
```

## Configuration

Default configuration in `config/default.yaml`:

```yaml
vsa:
  dim: 10000
  seed: 42

lgnn:
  coord_dim: 2
  hidden_dim: 64
  num_layers: 3

training:
  num_episodes: 1000
  steps_per_episode: 500
  lr: 0.001
  batch_size: 32

language:
  vsa_dim: 10000
  temperature: 0.07
```

## Testing

```bash
# All tests
python -m pytest tests/ -v

# Specific modules
python -m pytest tests/test_lgnn.py -v     # LGNN physics
python -m pytest tests/test_language.py -v  # Language layer
python -m pytest tests/test_lnn.py -v       # LNN
```

## Development Principles

- **Physics first**: conservation laws and causal structure are hardcoded in network architecture, not learned from data
- **Recipe, not database**: VSA stores relationship patterns (attribute IDs), not raw states
- **On-demand synthesis**: concept vectors are computed from recipes when needed, not stored permanently
- **Cross-modal grounding**: language is anchored to physical experience through contrastive alignment
- **Modular and testable**: every component has isolated tests and clear interfaces

## References

- [1] Hwang et al., "Learning the Dynamics of Particle-based Systems with Lagrangian Graph Neural Networks", 2023
- [2] Cranmer et al., "Lagrangian Neural Networks", 2020
- [3] Frady et al., "Vector Symbolic Architectures as a Computing Framework for Emerging Hardware", 2023
- [4] Kanerva, "Hyperdimensional Computing", 2009

## License

This project is open for academic and research purposes.
