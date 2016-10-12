#!/usr/bin/env python

import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='mesaPlot',
      version='0.0',
      description='Library for reading and plotting MESA data',
      license="GPLv2+",
      long_description=read('README.md'),
      author='Robert Farmer',
      author_email='rjfarmer@asu.edu',
      url='https://github.com/rjfarmer/mesaPlot',
      packages=["mesaPlot"],
      classifiers=[
			"Development Status :: 1 - Planning",
			"Intended Audience :: Science/Research",
			"License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
			"Topic :: Scientific/Engineering :: Astronomy",
      ]
     )
