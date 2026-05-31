"""PHYSMOL build script - builds C extensions via pybind11.

C extensions are optional - the package works without them using numpy fallback.
"""

import os
import sys
from setuptools import setup, find_packages

# Source directories
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(SRC_DIR, "src", "core")
BIND_DIR = os.path.join(SRC_DIR, "src", "bindings")

# Try to build C extensions (optional)
ext_modules = []
cmdclass = {}

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext

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
    cmdclass = {"build_ext": build_ext}
except ImportError:
    print("pybind11 not found - building without C extensions (numpy fallback)")
except Exception as e:
    print(f"Failed to setup C extensions: {e}")
    print("Building without C extensions (numpy fallback)")

setup(
    name="physmol",
    version="0.2.0",
    description="PHYSMOL: Physical Isomorphism and Symbolic Binding for Embodied Concept Learning",
    author="PHYSMOL Team",
    package_dir={"": "src/python"},
    packages=find_packages(where="src/python"),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
    ],
    extras_require={
        "pybind11": ["pybind11>=2.10"],
        "mujoco": ["mujoco>=2.3"],
        "torch": ["torch>=2.0"],
        "modelscope": ["modelscope>=1.9"],
        "huggingface": ["datasets"],
        "dev": ["pytest", "pytest-benchmark"],
    },
)
