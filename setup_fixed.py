# Fixed setup.py
from setuptools import setup, find_packages
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Simple setup without README reading
ext_modules = [
    Pybind11Extension(
        "pipeline_sim",
        ["python/src/bindings.cpp"],
        include_dirs=[
            "core/include",
            pybind11.get_include(),
        ],
        cxx_std=17,
    ),
]

setup(
    name="pipeline-sim",
    version="0.1.0",
    description="Pipeline simulation system",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    python_requires=">=3.7",
    zip_safe=False,
)
