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

sources = [
    {'name': 'grid_001',
    'source': r"""
kernel void grid_001(global int *data) {
    for(int i = 0; i < 1000000; i++) {
    }
}
""", 'grid': 1
},
    {'name': 'grid_008',
    'source': r"""
kernel void grid_008(global int *data) {
    for(int i = 0; i < 1000000; i++) {
    }
}
""", 'grid': 8
},
    {'name': 'grid_016',
    'source': r"""
kernel void grid_016(global int *data) {
    for(int i = 0; i < 1000000; i++) {
    }
}
""", 'grid': 16
},
    {'name': 'grid_032',
    'source': r"""
kernel void grid_032(global int *data) {
    for(int i = 0; i < 1000000; i++) {
    }
}
""", 'grid': 32
},
    {'name': 'grid_048',
    'source': r"""
kernel void grid_048(global int *data) {
    for(int i = 0; i < 1000000; i++) {
    }
}
""", 'grid': 48
},
    {'name': 'grid_064',
    'source': r"""
kernel void grid_064(global int *data) {
    for(int i = 0; i < 1000000; i++) {
    }
}
""", 'grid': 64
},
    {'name': 'grid_128',
    'source': r"""
kernel void grid_128(global int *data) {
    for(int i = 0; i < 1000000; i++) {
    }
}
""", 'grid': 128
}
]

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

def timeKernel(name, grid_x, kernel):
    # clearComputeCache()
    grid = (grid_x,1,1)
    block = (32,1,1)
    q.finish()
    inittime()
    call_cl_kernel(kernel, q, grid, block, d_cl)
    q.finish()
    return timecheck(name)
    # print(getPtx('mykernel'))

times = {}
for info in sources:
    clearComputeCache()
    name = info['name']
    source = info['source']
    grid = info['grid']
    kernel = buildKernel(name, source)
    print('built kernel')
    for it in range(3):
        t = timeKernel(name, grid, kernel)
        times[name] = t
    print(getPtx(name))
for name, time in sorted(times.items()):
    print(name, time)

