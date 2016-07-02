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

gpu_idx = 0

platforms = cl.get_platforms()
i = 0
for platform in platforms:
   gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
   if gpu_idx < i + len(gpu_devices):
       ctx = cl.Context(devices=[gpu_devices[gpu_idx-i]])
       break
   i += len(gpu_devices)

print('context', ctx)
q = cl.CommandQueue(ctx)

mf = cl.mem_flags

sources = {
    'kernel_for_loop_1e6': r"""
kernel void kernel_for_loop_1e6(global int *data) {
    for(int i = 0; i < 1000000; i++) {
    }
}
"""
,
    'kernel_for_loop_1e5': r"""
kernel void kernel_for_loop_1e5(global int *data) {
    for(int i = 0; i < 100000; i++) {
    }
}
"""
,
    'kernel_for_loop_1e4': r"""
kernel void kernel_for_loop_1e4(global int *data) {
    for(int i = 0; i < 10000; i++) {
    }
}
"""
,
    'kernel_for_loop_1e5_sum': r"""
kernel void kernel_for_loop_1e5_sum(global int *data) {
    int res = 0;
    for(int i = 0; i < 100000; i++) {
        res += i;
    }
}
"""
,
    'kernel_for_loop_1e5_sum_const': r"""
kernel void kernel_for_loop_1e5_sum_const(global int *data) {
    int res = 0;
    for(int i = 0; i < 100000; i++) {
        res += 15;
    }
}
"""
,
    'kernel_for_loop_1e5_mul_const': r"""
kernel void kernel_for_loop_1e5_mul_const(global int *data) {
    int res = 0;
    for(int i = 0; i < 100000; i++) {
        res *= 15;
    }
}
"""
,
    'kernel_for_loop_1e5_div_const': r"""
kernel void kernel_for_loop_1e5_div_const(global int *data) {
    int res = 0;
    for(int i = 0; i < 100000; i++) {
        res /= 15;
    }
}
"""
,
    'kernel_for_loop_1e5_float_div_const': r"""
kernel void kernel_for_loop_1e5_float_div_const(global int *data) {
    float res = 0;
    for(int i = 0; i < 100000; i++) {
        res /= 15.0f;
    }
}
"""
,
    'kernel_for_loop_1e5_float_mul_const': r"""
kernel void kernel_for_loop_1e5_float_mul_const(global int *data) {
    float res = 0;
    for(int i = 0; i < 100000; i++) {
        res *= 15.0f;
    }
}
"""
,
    'kernel_for_loop_1e5_float_add_const': r"""
kernel void kernel_for_loop_1e5_float_add_const(global int *data) {
    float res = 0;
    for(int i = 0; i < 100000; i++) {
        res += 15.0f;
    }
}
"""
}

optimized = set()

def clearComputeCache():
    cache_dir = join(os.environ['HOME'], '.nv/ComputeCache')
    for subdir in os.listdir(cache_dir):
        if subdir == 'index':
            continue
        print('clean', subdir)
        subprocess.call(['rm', '-Rf', join(cache_dir, subdir)])
#    subprocess.call(['rm', '-Rf', join(os.environ['HOME'], '.nv/ComputeCache')])

def getPtx(kernelName):
    with open('/tmp/gpucmd.sh', 'w') as f:
        f.write(r"""#!/bin/bash
        cat $(grep -r %s ~/.nv/ComputeCache | awk '{print $3}')
"""  % kernelName)
    filepath = subprocess.check_output(['/bin/bash', '/tmp/gpucmd.sh'])
    filepath_utf8 = ''
    for byte in filepath:
        # print(byte)
        if byte >= 10 and byte < 128:
           if chr(byte) in string.printable:
               filepath_utf8 += chr(byte)
    # print('filepath', filepath)
    #print(kernelName)
    print(filepath_utf8.split('--opt-level')[0])

def buildKernel(name, source):
    options = '-cl-opt-disable'
    if name in optimized:
        print('ENABLING OPTIMIZATIONS')
        options = ''
    return cl.Program(ctx, source).build(options=options).__getattr__(name) 

d = np.zeros((1024*1024 * 32 * 2,), dtype=np.float32)
d_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=d)

def timeKernel(name, kernel):
    # clearComputeCache()
    grid = (1,1,1)  # just one workgroup
    block = (32,1,1)
    q.finish()
    inittime()
    call_cl_kernel(kernel, q, grid, block, d_cl)
    q.finish()
    return timecheck(name)
    # print(getPtx('mykernel'))

times = {}
for name, source in sorted(sources.items()):
    clearComputeCache()
    kernel = buildKernel(name, source)
    print('built kernel')
    for it in range(3):
        t = timeKernel(name, kernel)
        times[name] = t
    print(getPtx(name))
for name, time in sorted(times.items()):
    print(name, time)

