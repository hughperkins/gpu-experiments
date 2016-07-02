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
    'kernel_store_to_local': r"""
kernel void kernel_store_to_local(global int *data) {
    local int F[32];
    F[0] = 123;
}
"""
,    'kernel_init_local': r"""
kernel void kernel_init_local(global int *data) {
    local int F[32];
    for(int i = 0; i < 32; i++) {
       F[i] = 0;
    };
}
"""
,    'kernel_init_local_noloop': r"""
kernel void kernel_init_local_noloop(global int *data) {
    local int F[32];
    F[get_local_id(0)] = 0;
}
"""
,    'kernel_copy_local_to_global': r"""
kernel void kernel_copy_local_to_global(global int *data) {
    local int F[32];
    int tid = get_local_id(0);
    data[tid] = F[tid];
}
"""
,    'kernel_copy_local_from_global': r"""
kernel void kernel_copy_local_from_global(global int *data) {
    local int F[32];
    int tid = get_local_id(0);
    F[tid] = data[tid];
}
"""
,    'kernel_copy_local_to_global_gid': r"""
kernel void kernel_copy_local_to_global_gid(global int *data) {
    local int F[32];
    int tid = get_local_id(0);
    int gid = get_global_id(0);
    data[gid] = F[tid];
}
"""
,    'kernel_copy_local_from_global_gid': r"""
kernel void kernel_copy_local_from_global_gid(global int *data) {
    local int F[32];
    int tid = get_local_id(0);
    int gid = get_global_id(0);
    F[tid] = data[gid];
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

