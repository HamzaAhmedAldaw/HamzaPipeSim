# AI_GENERATED: Python package setup
from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "pipeline_sim._core",
        ["src/bindings.cpp"],
        include_dirs=["../core/include"],
        libraries=["pipeline_sim_core"],
        library_dirs=["../build/core"],
        language="c++",
        cxx_std=17,
    ),
]

setup(
    name="pipeline-sim",
    version="0.1.0",
    author="Pipeline-Sim Contributors",
    description="Next-generation petroleum pipeline simulation",
    long_description=open("../README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19",
        "pandas>=1.2",
        "matplotlib>=3.3",
        "scipy>=1.6",
        "pyyaml>=5.4",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.9",
            "sphinx>=4.0",
        ],
        "ml": [
            "scikit-learn>=0.24",
            "tensorflow>=2.6",
            "torch>=1.9",
        ],
    },
)