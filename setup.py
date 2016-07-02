#!/usr/bin/env python

import os
from setuptools import setup, find_packages, Command
import subprocess

setup(name='gpu-experiments',
      version='SNAPSHOT',
      description="GPU Experiments",
      long_description=open('README.md').read(),
      author='Hugh Perkins',
      author_email='hughperkins@gmail.com',
      url='http://github.com/hughperkins/gpu-experiments',
      # license='License :: OSI Approved :: Apache Software License',
      scripts=[],
      packages=find_packages(exclude=["tests"]),
      package_data={'gpuexperiments': []}
)
      #classifiers=['Development Status :: 2 - Beta',
      #             'Intended Audience :: Developers',
      #             'Intended Audience :: Science/Research',
      #             'License :: OSI Approved :: Apache Software License',
      #             'Operating System :: POSIX',
      #             'Operating System :: MacOS :: MacOS X',
      #             'Programming Language :: Python',
      #             'Topic :: Scientific/Engineering :: ' +
      #             'Artificial Intelligence',
      #             'Topic :: Scientific/Engineering :: Information Analysis',
      #             'Topic :: System :: Distributed Computing'])

