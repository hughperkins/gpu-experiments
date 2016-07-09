# Note that this will erase your nvidia cache, ~/.nv/ComputeCache  This may or may not be an undesirable side-effect for you.  For example, cutorch will take 1-2 minutes or so to start after this cache has been emptied.

from __future__ import print_function, division
import time
import string
import random
import numpy as np
import pyopencl as cl
import subprocess
import os
from os.path import join
from gpuexperiments.callkernel import call_cl_kernel
#import gpuexperiments.cpu_check
from gpuexperiments.timecheck import inittime, timecheck
from lib_clgpuexp import clearComputeCache, getPtx, timeKernel, buildKernel, initClGpu


initClGpu()

sources = {
    'kernel01': r"""
static inline void add_one(private int *a) {
   *a += 1;
}

kernel void kernel01(global int *data, global int *out) {
    int b = 4;
    add_one(&b);
    data[0] = b;
}
"""
,
    'kernel02': r"""
inline void add_one(private int *a) {
   *a += 1;
}

kernel void kernel02(global int *data, global int *out) {
    int b = 4;
    add_one(&b);
    data[0] = b;
}
"""
,
    'kernel03': r"""
#define add_one(a) \
   a += 1;

kernel void kernel03(global int *data, global int *out) {
    int b = 4;
    add_one(b);
    data[0] = b;
}
"""
,
    'kernel04': r"""
#define add_one(a) \
   a += 1;

kernel void kernel04(global int *data, global int *out) {
    int b = 4;
    add_one(b);
    data[0] = b;
}
"""
,
    'kernel05': r"""
static inline void add_one(private int *a) {
   *a += 1;
}

kernel void kernel05(global int *data, global int *out) {
    int b = 4;
    add_one(&b);
    data[0] = b;
}
"""
}

optimized = set()
optimized.add('kernel04')
optimized.add('kernel05')

times = {}
for name, source in sorted(sources.items()):
    clearComputeCache()
    options = '' if name in optimized else '-cl-opt-disable'
    kernel = buildKernel(name, source, options=options)
    print('built kernel')
    for it in range(3):
        t = timeKernel(name, kernel, grid_x=1024*1024, block_x=32)
        times[name] = t
    print(getPtx(name))

print('name\t\ttot ms')
for name, time in sorted(times.items()):
    print('%s\t%.1f' % (name, time))

