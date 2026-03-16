"""
Build script for Cython-accelerated backtest module.

Usage:
    python src/setup_cython.py build_ext --inplace
"""
import os
import sys
from setuptools import setup, Extension

# Try to import Cython; if unavailable, skip
try:
    from Cython.Build import cythonize
    import numpy as np

    extensions = [
        Extension(
            "src.fast_backtest",
            [os.path.join("src", "fast_backtest.pyx")],
            include_dirs=[np.get_include()],
        )
    ]

    setup(
        name="fast_backtest",
        ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    )

except ImportError:
    print("Cython or NumPy not available. Skipping Cython build.")
    print("The system will use the pure Python fallback.")
    sys.exit(0)
