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
    'kernel01': r"""
static inline void add_one(private int *a) {
   *a += 1;
}

kernel void kernel01(global int *data) {
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

kernel void kernel02(global int *data) {
    int b = 4;
    add_one(&b);
    data[0] = b;
}
"""
,
    'kernel03': r"""
#define add_one(a) \
   a += 1;

kernel void kernel03(global int *data) {
    int b = 4;
    add_one(b);
    data[0] = b;
}
"""
,
    'kernel04': r"""
#define add_one(a) \
   a += 1;

kernel void kernel04(global int *data) {
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

kernel void kernel05(global int *data) {
    int b = 4;
    add_one(&b);
    data[0] = b;
}
"""
}

optimized = set()
optimized.add('kernel04')
optimized.add('kernel05')

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
    grid = (1024*1024,1,1)
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

