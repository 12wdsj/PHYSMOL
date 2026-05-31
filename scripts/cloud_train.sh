#!/bin/bash
# PHYSMOL Cloud Training Script
# Usage: bash scripts/cloud_train.sh [all|phase1|phase2|phase3|abstract]

set -euo pipefail

# Configuration
PROJECT_DIR="${PROJECT_DIR:-/mnt/workspace/PHYSMOL}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints}"
DEVICE="${DEVICE:-auto}"  # auto, cuda, rocm, cpu
VSA_DIM="${VSA_DIM:-4096}"
EPOCHS="${EPOCHS:-500}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------

setup_environment() {
    log "Setting up environment..."

    cd "$PROJECT_DIR"

    # Install Python dependencies
    pip install -e .

    # Install optional dependencies
    pip install sentence-transformers jieba 2>/dev/null || warn "Some optional deps failed"

    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        log "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    elif command -v rocm-smi &> /dev/null; then
        log "AMD GPU detected:"
        rocm-smi --showproductname
    else
        warn "No GPU detected, using CPU"
        DEVICE="cpu"
    fi

    mkdir -p "$CHECKPOINT_DIR"
}

# ------------------------------------------------------------------
# Download pre-trained vectors (with mirror fallback)
# ------------------------------------------------------------------

download_vectors() {
    log "Downloading pre-trained word vectors..."
    mkdir -p ./vectors

    # Check if vectors already exist
    if [[ -f "./vectors/cc.zh.300.vec" ]] && [[ -f "./vectors/cc.en.300.vec" ]]; then
        log "Vectors already exist, skipping download."
        return 0
    fi

    # Try multiple mirrors
    MIRRORS=(
        "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl"
        "https://mirror.ghproxy.com/https://github.com/facebookresearch/fastText/raw/main/vectors/crawl"
        "https://hf-mirror.com/datasets/facebook/fasttext/resolve/main/vectors-crawl"
    )

    for lang in zh en; do
        if [[ -f "./vectors/cc.${lang}.300.vec" ]]; then
            log "cc.${lang}.300.vec already exists, skipping."
            continue
        fi

        log "Downloading cc.${lang}.300.vec..."
        downloaded=0

        for mirror in "${MIRRORS[@]}"; do
            url="${mirror}/cc.${lang}.300.vec.gz"
            log "Trying: $url"

            if command -v wget &> /dev/null; then
                wget -q --timeout=30 --tries=2 -O "./vectors/cc.${lang}.300.vec.gz" "$url" 2>/dev/null
            elif command -v curl &> /dev/null; then
                curl -sL --connect-timeout 30 --retry 2 -o "./vectors/cc.${lang}.300.vec.gz" "$url" 2>/dev/null
            else
                error "Neither wget nor curl found. Install one."
            fi

            if [[ $? -eq 0 ]] && [[ -s "./vectors/cc.${lang}.300.vec.gz" ]]; then
                log "Downloaded, extracting..."
                gunzip -f "./vectors/cc.${lang}.300.vec.gz"
                downloaded=1
                log "cc.${lang}.300.vec ready."
                break
            else
                warn "Failed from $mirror, trying next..."
                rm -f "./vectors/cc.${lang}.300.vec.gz"
            fi
        done

        if [[ $downloaded -eq 0 ]]; then
            warn "All mirrors failed for cc.${lang}.300.vec"
            warn "You can manually download from: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.${lang}.300.vec.gz"
            warn "Or use sentence-transformers instead (no download needed)"
        fi
    done

    log "Vector download complete."
}

# ------------------------------------------------------------------
# Phase 1: Physics Learning
# ------------------------------------------------------------------

train_phase1() {
    log "Phase 1: Physics Learning (LGNN)"

    python3 -m physmol.unified_train \
        --phase 1 \
        --device "$DEVICE" \
        --epochs "$EPOCHS" \
        --save-path "$CHECKPOINT_DIR/phase1"

    log "Phase 1 complete. Checkpoints: $CHECKPOINT_DIR/phase1"
}

# ------------------------------------------------------------------
# Phase 2: Concept Formation
# ------------------------------------------------------------------

train_phase2() {
    log "Phase 2: Concept Formation"

    python3 -m physmol.unified_train \
        --phase 2 \
        --save-path "$CHECKPOINT_DIR/phase2"

    log "Phase 2 complete."
}

# ------------------------------------------------------------------
# Phase 3: Language Alignment
# ------------------------------------------------------------------

train_phase3() {
    log "Phase 3: Language Alignment"

    python3 -m physmol.unified_train \
        --phase 3 \
        --device "$DEVICE" \
        --phase3-epochs 100 \
        --save-path "$CHECKPOINT_DIR/phase3"

    log "Phase 3 complete."
}

# ------------------------------------------------------------------
# Abstract Cognition Training
# ------------------------------------------------------------------

train_abstract() {
    log "Abstract Cognition Training"

    # Use built-in seed data
    python3 -m physmol.abstract_train \
        --use-builtin \
        --out-dir "$CHECKPOINT_DIR/abstract"

    log "Abstract training complete."
}

# ------------------------------------------------------------------
# Enhanced Language Training
# ------------------------------------------------------------------

train_language() {
    log "Enhanced Language Training"

    python3 << 'EOF'
import sys
import os
sys.path.insert(0, "src/python")

from physmol.language.enhanced_encoder import EnhancedTextEncoder
from physmol.language.cognitive import CognitiveInterface

print("=" * 60)
print("Enhanced Language Training")
print("=" * 60)

# Initialize encoder
encoder = EnhancedTextEncoder(vsa_dim=4096)

# Load pre-trained vectors if available
vector_paths = [
    "./vectors/cc.zh.300.vec",
    "./vectors/cc.en.300.vec",
]

for path in vector_paths:
    if os.path.exists(path):
        lang = "Chinese" if "zh" in path else "English"
        count = encoder.load_pretrained_vectors(path, max_words=50000)
        print(f"Loaded {count} {lang} vectors from {path}")

# Initialize sentence-transformer
if encoder.init_sentence_transformer("paraphrase-multilingual-MiniLM-L12-v2"):
    print("Sentence-transformer loaded successfully")
else:
    print("Using word-level encoding (install sentence-transformers for better quality)")

# Test encoding
test_texts = [
    "这是一个测试",
    "This is a test",
    "快速排序算法",
    "Quicksort algorithm",
    "物理模拟",
    "Physics simulation",
]

print("\nEncoding test:")
for text in test_texts:
    vec = encoder.encode(text)
    print(f"  '{text}' -> dim={len(vec)}, norm={np.linalg.norm(vec):.4f}")

# Save vocabulary
encoder.save_vocabulary("./checkpoints/vocabulary.json")
print(f"\nVocabulary saved to ./checkpoints/vocabulary.json")
print(f"Stats: {encoder.get_stats()}")
EOF
}

# ------------------------------------------------------------------
# Broca Language Production Training
# ------------------------------------------------------------------

train_broca() {
    log "Broca Language Production Training"

    # Train from ModelScope dataset
    python3 -m physmol.dialogue_trainer \
        --modelscope BelleGroup/train_1M_CN \
        --limit 10000 \
        --epochs 3 \
        --out-dir "$CHECKPOINT_DIR/broca"

    log "Broca training complete."
}

# ------------------------------------------------------------------
# Full pipeline
# ------------------------------------------------------------------

train_all() {
    log "Starting full PHYSMOL training pipeline"

    setup_environment
    download_vectors
    train_phase1
    train_phase2
    train_phase3
    train_abstract
    train_language
    train_broca

    log "All training complete!"
    log "Checkpoints: $CHECKPOINT_DIR"
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

case "${1:-all}" in
    all)
        train_all
        ;;
    setup)
        setup_environment
        ;;
    vectors)
        download_vectors
        ;;
    phase1)
        setup_environment
        train_phase1
        ;;
    phase2)
        setup_environment
        train_phase2
        ;;
    phase3)
        setup_environment
        train_phase3
        ;;
    abstract)
        setup_environment
        train_abstract
        ;;
    language)
        setup_environment
        train_language
        ;;
    broca)
        setup_environment
        train_broca
        ;;
    *)
        echo "Usage: $0 {all|setup|vectors|phase1|phase2|phase3|abstract|language|broca}"
        echo ""
        echo "Environment variables:"
        echo "  PROJECT_DIR    - Project directory (default: /mnt/workspace/PHYSMOL)"
        echo "  CHECKPOINT_DIR - Checkpoint directory (default: \$PROJECT_DIR/checkpoints)"
        echo "  DEVICE         - Device: auto, cuda, rocm, cpu (default: auto)"
        echo "  EPOCHS         - Training epochs (default: 500)"
        echo ""
        echo "Examples:"
        echo "  # Full training on GPU"
        echo "  DEVICE=cuda bash scripts/cloud_train.sh all"
        echo ""
        echo "  # Only physics training"
        echo "  DEVICE=cuda bash scripts/cloud_train.sh phase1"
        echo ""
        echo "  # Only language training"
        echo "  bash scripts/cloud_train.sh language"
        exit 1
        ;;
esac
