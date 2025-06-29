from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

# Simple extension - only include what we need
ext_modules = [
    Pybind11Extension(
        "pipeline_sim",
        sources=[
            # Only the essential files
            "core/src/node.cpp",
            "core/src/pipe.cpp",
            "core/src/network.cpp",
            "core/src/fluid_properties.cpp",
            "python/src/bindings.cpp",
        ],
        include_dirs=[
            "core/include",
            pybind11.get_include(),
            pybind11.get_include(user=True),
            r"C:\vcpkg\installed\x64-windows\include",
            r"C:\vcpkg\installed\x64-windows\include\eigen3",
        ],
        define_macros=[("NOMINMAX", None)],
        language='c++',
        cxx_std=17,
    ),
]

# Windows compiler settings
if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        for ext in ext_modules:
            ext.extra_compile_args = ["/EHsc", "/bigobj", "/std:c++17"]

    setup(
        name="pipeline-sim",
        version="0.1.0",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        zip_safe=False,
        python_requires=">=3.7",
    )