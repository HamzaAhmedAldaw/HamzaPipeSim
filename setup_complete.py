from setuptools import setup, find_packages
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
import glob

# Find all source files
source_files = ["python/src/bindings.cpp"]
source_files.extend(glob.glob("core/src/*.cpp"))

ext_modules = [
    Pybind11Extension(
        "pipeline_sim",
        source_files,
        include_dirs=[
            "core/include",
            pybind11.get_include(),
            r"C:\vcpkg\installed\x64-windows\include",
            r"C:\vcpkg\installed\x64-windows\include\eigen3"
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
