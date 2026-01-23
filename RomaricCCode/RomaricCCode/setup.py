from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "saliency",                        # must match PYBIND11_MODULE name
        ["bindings.cpp"],                  # your binding source
        cxx_std=17,                        # needed for your code
    ),
]

setup(
    name="saliency",
    version="0.1.0",
    author="Your Name",
    description="Event-based saliency detection (pybind11)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
