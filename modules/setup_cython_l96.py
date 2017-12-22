#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from distutils.core import setup
from Cython.Build import cythonize

setup(name="lorenz96_cython", ext_modules=cythonize('/scratch/uni/u234/u234069/lorenz96_lectureedition/modules/lorenz96_cython.pyx'))