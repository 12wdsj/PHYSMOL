"""PHYSMOL: Physical Isomorphism and Symbolic Binding for Embodied Concept Learning."""

__version__ = "0.2.0"

# Always-available components (numpy-only, no C extensions needed)
from .world_model import HierarchicalWorldModel, CuriositySignal
from .motivation import IntrinsicMotivationSystem
from .long_term_memory import LongTermMemory, MemoryRecord
from .transfer import CrossDomainTransferEngine, Domain, TransferSchema
from .progress import ProgressLogger
from .training_data import (
    TrainingExample, ModelScopeDatasetAdapter, LocalDatasetAdapter,
    PHYSMOLTrainingDataBuilder,
)
from .abstract_training import (
    AbstractCognitionTrainer, ProofRuleLibrary, LegalCaseBase,
    ValueConstraintLearner,
)
from .perception import (
    MultiModalPerception, VisionEncoder, AudioEncoder,
    TactileEncoder, OlfactoryEncoder, ProprioceptiveEncoder
)
from .lnn import LagrangianNetwork
from .knowledge_acquisition import KnowledgeAcquisition
from .dialogue_trainer import DialogueTrainer
from .continuous_learning import ContinuousLearner

# VSA Recipe Store (correct philosophy: recipe-based, not database)
from .vsa_store import AttributePrimitivePool, RecipeStore, ConceptSynthesizer

# Language Cognitive Layer
from .language import (
    CognitiveInterface, TextToVSA, SemanticParser, ReasoningEngine, Responder,
    VSALanguageGenerator, AbstractConceptReasoner, TheoryOfMindModel,
    DialogueState, AbstractTaskReasoner,
)

# Enhanced language encoder (optional, for large vocabulary support)
try:
    from .language.enhanced_encoder import EnhancedTextEncoder
except ImportError:
    pass

# Unified Training Pipeline (requires PyTorch LGNN dependencies)
try:
    from .unified_train import UnifiedTrainer
except ImportError:
    pass

# LGNN (requires PyTorch)
try:
    from .lgnn import LagrangianGraphNetwork, PhysicsGraph
    from .lgnn_train import LGNNTrainer, TrajectoryDataset
except ImportError:
    pass

# Check C extension availability
_HAS_C_EXT = False
try:
    from . import _vsa
    from . import _snn
    _HAS_C_EXT = True
except ImportError:
    pass


def check_build():
    """Check if C extensions are built."""
    if not _HAS_C_EXT:
        print("WARNING: C extensions not built. Run: python setup.py build_ext --inplace")
        print("  or: make build")
        print("  VSA and SNN will not work without C extensions.")
        print("  LNN, perception, and world_model work fine without them.")
        return False
    return True
