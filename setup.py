#!/usr/bin/env python

import os
from setuptools import setup
import unittest

def my_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite

setup(name='mesaPlot',
      version='0.0',
      description='Library for reading and plotting MESA data',
      license="GPLv2+",
      author='Robert Farmer',
      author_email='r.j.farmer@uva.nl',
      url='https://github.com/rjfarmer/mesaPlot',
      packages=["mesaPlot"],
      tests_require=["unittest2"],
      test_suite='setup.my_test_suite',
      classifiers=[
			"Development Status :: 5 - Production/Stable",
			"Intended Audience :: Science/Research",
			"License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
			"Topic :: Scientific/Engineering :: Astronomy",
      ],
      extras_require={
		'dev': [
			'unittest2'
			]
		}
     )
