#!/usr/bin/env python

import os
from setuptools import setup

setup(name='mesaPlot',
      version='0.0',
      description='Library for reading and plotting MESA data',
      license="GPLv2+",
      author='Robert Farmer',
      author_email='r.j.farmer@uva.nl',
      url='https://github.com/rjfarmer/mesaPlot',
      packages=["mesaPlot"],
      classifiers=[
			"Development Status :: 5 - Production/Stable",
			"Intended Audience :: Science/Research",
			"License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
			"Topic :: Scientific/Engineering :: Astronomy",
      ]
     )
