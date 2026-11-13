"""Optional C++ speed-up (reachlens._fast). If pybind11, a compiler or the MuJoCo headers are missing, the build skips
it and reachlens runs on the pure Python solver."""
import os

from setuptools import setup

ext_modules = []
try:
    import mujoco
    import pybind11
    from setuptools import Extension

    mj_dir = os.path.dirname(mujoco.__file__)
    libs = [f for f in os.listdir(mj_dir) if f.startswith("libmujoco.so")]
    if libs and os.path.isdir(os.path.join(mj_dir, "include")):
        ext_modules = [Extension(
            "reachlens._fast", ["cpp/fast_ik.cpp"],
            include_dirs=[pybind11.get_include(), os.path.join(mj_dir, "include")],
            library_dirs=[mj_dir], libraries=[":" + sorted(libs)[0]],
            runtime_library_dirs=[mj_dir], extra_compile_args=["-O3", "-std=c++17"], optional=True)]
except Exception:
    pass

setup(ext_modules=ext_modules)
