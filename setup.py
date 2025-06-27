# ===== setup.py (root) =====
"""
Pipeline-Sim Setup Script

AI_GENERATED: Package configuration for distribution
"""
from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pipeline-sim",
    version="0.1.0",
    author="Pipeline-Sim Contributors",
    author_email="",
    description="Next-generation petroleum pipeline simulation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pipeline-sim/pipeline-sim",
    project_urls={
        "Bug Tracker": "https://github.com/pipeline-sim/pipeline-sim/issues",
        "Documentation": "https://pipeline-sim.readthedocs.io",
        "Source Code": "https://github.com/pipeline-sim/pipeline-sim",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: C++",
    ],
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.9",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "ml": [
            "scikit-learn>=0.24",
            "tensorflow>=2.6",
            "torch>=1.9",
        ],
        "gui": [
            "matplotlib>=3.3",
            "networkx>=2.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "pipeline-sim=tools.cli.pipeline_sim_cli:cli",
            "pipeline-sim-gui=tools.gui.pipeline_sim_gui:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
