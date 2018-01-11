#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from distutils.core import setup
from Cython.Build import cythonize

setup(name="lorenz96_cython", ext_modules=cythonize('C:\\Users\Sebastian\Github\lorenz96_lectureedition\modules\lorenz96_cython.pyx'))