# Fixed setup.py with Eigen path
from setuptools import setup, find_packages
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "pipeline_sim",
        ["python/src/bindings.cpp"],
        include_dirs=[
            "core/include",
            pybind11.get_include(),
            r"C:\vcpkg\installed\x64-windows\include",  # Eigen is here
            r"C:\vcpkg\installed\x64-windows\include\eigen3"  # Or here
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
