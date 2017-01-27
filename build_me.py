# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 09:38:04 2013

@author: Raman
"""

import logging
import numpy
logging.basicConfig(level=logging.DEBUG)

from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = False

setup(
    cmdclass = {'build_ext': build_ext},
    include_dirs=[numpy.get_include()],
    ext_modules = [Extension("fast_cubic_spline",
                             sources=["fast_cubic_spline.pyx"])
                  ],
    script_args = ['build_ext', '--inplace']
)