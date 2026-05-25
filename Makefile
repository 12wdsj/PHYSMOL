.PHONY: build clean test install dev

# Build C extensions in-place
build:
	pip install pybind11 numpy
	python setup.py build_ext --inplace

# Development install
dev: build
	pip install -e ".[dev]"

# Run C unit tests
test-c:
	gcc -O3 -mavx2 -mfma -o test_vsa tests/test_vsa.c src/core/vsa.c -lm && ./test_vsa
	gcc -O3 -mavx2 -mfma -o test_snn tests/test_snn.c src/core/snn.c src/core/causal.c -lm && ./test_snn

# Run Python tests
test: build
	python -m pytest tests/ -v

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info/
	rm -f src/python/physmol/_vsa*.so src/python/physmol/_snn*.so src/python/physmol/_lnn*.so
	rm -f test_vsa test_snn
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Install dependencies
deps:
	pip install pybind11 numpy mujoco pytest

# Quick smoke test
smoke: build
	python -c "from physmol import check_build; check_build(); print('OK')"
	python -c "from physmol.vsa import VectorSymbolicArchitecture; vsa=VectorSymbolicArchitecture(4096); a=vsa.random_bipolar(); b=vsa.random_bipolar(); c=vsa.bind(a,b); print(f'VSA bind test: sim={vsa.similarity(a,c):.4f}')"
	python -c "from physmol.snn import SpikingNetwork; net=SpikingNetwork(512,512); print(f'SNN test: {net}')"
	python -c "from physmol.lnn import LagrangianNetwork; lnn=LagrangianNetwork(3,64); import numpy as np; q=np.array([1.0,0.0,0.0]); qd=np.array([0.1,0.0,0.0]); L=lnn.forward(q,qd); print(f'LNN test: L={L:.4f}')"
