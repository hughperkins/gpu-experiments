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
kernel void kernel01(global int *data) {
    local int F[32];
    F[0] = 123;
}
"""
,
    'kernel_int_nop': r"""
kernel void kernel_int_nop(global int *data) {
    local int F[32];
    F[0] = 123;
    int a = F[0];
    F[0] = a;
}
"""
,
    'kernel_int_add': r"""
kernel void kernel_int_add(global int *data) {
    local int F[32];
    // F[0] = 123;
    int a = F[0];
    int b = F[1];
    F[0] = a + b;
}
"""
,
    'kernel_int_mul': r"""
kernel void kernel_int_mul(global int *data) {
    local int F[32];
    // F[0] = 123;
    int a = F[0];
    int b = F[1];
    F[0] = a * b;
}
"""
,
    'kernel_int_div': r"""
kernel void kernel_int_div(global int *data) {
    local int F[32];
    // F[0] = 123;
    int a = F[0];
    int b = F[1];
    F[0] = a / b;
}
"""
,
    'kernel_int_shift': r"""
kernel void kernel_int_shift(global int *data) {
    local int F[32];
    // F[0] = 123;
    int a = F[0];
    int b = F[1];
    F[0] = a << b;
}
"""
,
    'kernel_float_nop': r"""
kernel void kernel_float_nop(global int *data) {
    local float F[32];
    float a = F[0];
    F[0] = a;
}
"""
,
    'kernel_float_add': r"""
kernel void kernel_float_add(global int *data) {
    local float F[32];
    // F[0] = 123;
    float a = F[0];
    float b = F[1];
    F[0] = a + b;
}
"""
,
    'kernel_float_mul': r"""
kernel void kernel_float_mul(global int *data) {
    local float F[32];
    // F[0] = 123;
    float a = F[0];
    float b = F[1];
    F[0] = a * b;
}
"""
,
    'kernel_float_div': r"""
kernel void kernel_float_div(global int *data) {
    local float F[32];
    // F[0] = 123;
    float a = F[0];
    float b = F[1];
    F[0] = a / b;
}
"""
}

optimized = set()

times = {}
for name, source in sorted(sources.items()):
    clearComputeCache()
    kernel = buildKernel(name, source, '' if name in optimized else '-cl-opt-disable')
    print('built kernel')
    for it in range(3):
        t = timeKernel(name, kernel)
        times[name] = t
    print(getPtx(name))
for name, time in sorted(times.items()):
    print(name, time)

