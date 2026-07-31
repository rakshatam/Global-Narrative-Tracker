import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

def ensure_windows_sdk():
    sdk_include = r"C:\Program Files (x86)\Windows Kits\10\Include"
    if not os.path.exists(sdk_include): return
    versions = [d for d in os.listdir(sdk_include) if d.startswith("10.")]
    if not versions: return
    latest = sorted(versions)[-1]
    sdk_base = r"C:\Program Files (x86)\Windows Kits\10"
    inc = [os.path.join(sdk_base,"Include",latest,d) for d in ("ucrt","shared","um")]
    lib = [os.path.join(sdk_base,"Lib",latest,d,"x64") for d in ("ucrt","um")]
    bn  = os.path.join(sdk_base,"bin",latest,"x64")
    os.environ["INCLUDE"] = ";".join(inc)+";"+os.environ.get("INCLUDE","")
    os.environ["LIB"]     = ";".join(lib)+";"+os.environ.get("LIB","")
    os.environ["PATH"]    = bn+";"+os.environ.get("PATH","")
    print(f"[setup.py] Injected Windows SDK {latest}")

ensure_windows_sdk()

extensions = [
    Extension(
        "lsh",
        sources=["lsh.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
        language="c++"
    )
]

setup(
    name="Cython LSH Module",
    ext_modules=cythonize(extensions, compiler_directives={'language_level' : "3"})
)
