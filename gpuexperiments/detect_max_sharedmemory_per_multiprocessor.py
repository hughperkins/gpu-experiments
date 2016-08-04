"""
This will create a single block, with shared memory,and increase the size of shared memory, till it crashes.  The
previous size of shared memory is thus the max.

Hmmm, actually thta wont work, we need multiple blocks in fact...

Hmmm, in fact, we cannot detect this by simply trying to get crashes, only by analyzing appropriate graphs...
(or checking the specs)
"""
from __future__ import print_function, division
import argparse
import time
import string
import random
import jinja2
import numpy as np
import pyopencl as cl
import subprocess
import os
from os.path import join
from gpuexperiments.callkernel import call_cl_kernel
# import gpuexperiments.cpu_check
from gpuexperiments.timecheck import inittime, timecheck
import lib_clgpuexp
from lib_clgpuexp import clearComputeCache, getPtx, timeKernel, buildKernel, initClGpu


initClGpu()

code_template = r"""
    kernel void {{kernelname}} (global float *data, global float *out, local float *F) {
    }
"""
name = 'test'
template = jinja2.Template(code_template, undefined=jinja2.StrictUndefined)
source = template.render(kernelname=name)
kernel = buildKernel(name, source)
print('built kernel')
grid = (1, 1, 1)
block = (1, 1, 1)
shared_bytes_kb = 1
while True:
    print('trying', shared_bytes_kb, 'kb')
    try:
        call_cl_kernel(kernel, lib_clgpuexp.q, grid, block, lib_clgpuexp.d_cl, lib_clgpuexp.out_cl, cl.LocalMemory(shared_bytes_kb * 1024 - 4))
        lib_clgpuexp.q.finish()
        shared_bytes_kb += 1
    except Exception as e:
        print('found limit')
        break
print('max shared memory per multiprocessor: %sKiB' % (shared_bytes_kb - 1))
