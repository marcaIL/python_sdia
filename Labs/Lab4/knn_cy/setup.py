import os
from distutils.core import setup
import numpy as np
from Cython.Build import cythonize


os.environ["CC"] = "gcc"


setup(
    ext_modules=cythonize("knn.pyx", annotate = True, language_level="3"),
    include_dirs=[np.get_include()],
)
