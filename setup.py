from setuptools import setup, find_packages
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import numpy

requirements = ['numpy']
revision = 0

ext_modules = []

ext_modules.append(Extension("fast_cubic_spline",
                            ["fast_cubic_spline.pyx"],
#                             libraries=["m"],
#                             extra_compile_args=['-fopenmp'],
#                             extra_link_args=['-fopenmp'],
                             )
                   )

setup(name='spline-interpolate',
      include_dirs=[numpy.get_include()],
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules, 
      version='0.1.dev%d'%revision,
      install_requires=requirements,
      packages=find_packages(),
      url='https://github.com/magnium/fast-cubic-spline-python'
      )
