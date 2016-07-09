from gpuexperiments.callkernel import call_cl_kernel
import subprocess
import os
import string
from os.path import join
import pyopencl as cl
import numpy as np
from gpuexperiments.timecheck import inittime, timecheck


ctx = None
q = None
mf = cl.mem_flags

d = None
d_cl = None

out = None
out_cl = None

def initClGpu(gpu_idx=0):
    global ctx, q, d, d_cl, out, out_cl

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

    d = np.zeros((1024*1024 * 32 * 2,), dtype=np.float32)
    d_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=d)

    out = np.zeros((1024,), dtype=np.float32)
    out_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=out)

    return ctx, q, mf

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
    print(filepath_utf8.split('--opt-level')[0].split('--reserve')[0])

def timeKernel(name, kernel, grid_x=1, block_x=32):
    # clearComputeCache()
    grid = (grid_x,1,1)
    block = (block_x,1,1)
    q.finish()
    inittime()
    call_cl_kernel(kernel, q, grid, block, d_cl, out_cl)
    q.finish()
    return timecheck(name)
    # print(getPtx('mykernel'))

def buildKernel(name, source, options=''):
    # options = '-cl-opt-disable'
    #if name in optimized:
    #    print('ENABLING OPTIMIZATIONS')
    #    options = ''
    return cl.Program(ctx, source).build(options=options).__getattr__(name) 

