"""PHYSMOL build script - builds C extensions via pybind11."""

import os
import sys
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

# Source directories
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(SRC_DIR, "src", "core")
BIND_DIR = os.path.join(SRC_DIR, "src", "bindings")

# Detect SIMD flags
extra_compile_args = ["-O3"]
if sys.platform != "win32":
    # Check for AVX2 support
    extra_compile_args.extend(["-mavx2", "-mfma"])

ext_modules = [
    Pybind11Extension(
        "physmol._vsa",
        [
            os.path.join(BIND_DIR, "vsa_py.cpp"),
            os.path.join(CORE_DIR, "vsa.c"),
        ],
        include_dirs=[SRC_DIR, CORE_DIR],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
    Pybind11Extension(
        "physmol._snn",
        [
            os.path.join(BIND_DIR, "snn_py.cpp"),
            os.path.join(CORE_DIR, "snn.c"),
            os.path.join(CORE_DIR, "causal.c"),
        ],
        include_dirs=[SRC_DIR, CORE_DIR],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
    Pybind11Extension(
        "physmol._lnn",
        [
            os.path.join(BIND_DIR, "lnn_py.cpp"),
            os.path.join(CORE_DIR, "lnn.c"),
            os.path.join(CORE_DIR, "memory.c"),
        ],
        include_dirs=[SRC_DIR, CORE_DIR],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
]

setup(
    name="physmol",
    version="0.1.0",
    description="PHYSMOL: Physical Isomorphism and Symbolic Binding for Embodied Concept Learning",
    author="PHYSMOL Team",
    package_dir={"": "src/python"},
    packages=find_packages(where="src/python"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "pybind11>=2.10",
    ],
    extras_require={
        "mujoco": ["mujoco>=2.3"],
        "torch": ["torch>=2.0"],
        "modelscope": ["modelscope>=1.9"],
        "dev": ["pytest", "pytest-benchmark"],
    },
)
