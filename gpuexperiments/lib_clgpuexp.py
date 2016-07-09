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
device = None
mf = cl.mem_flags

d = None
d_cl = None

out = None
out_cl = None

def initClGpu(gpu_idx=0):
    global ctx, q, device, d, d_cl, out, out_cl

    platforms = cl.get_platforms()
    i = 0
    for platform in platforms:
       gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
       if gpu_idx < i + len(gpu_devices):
           device = gpu_devices[gpu_idx-i]
           ctx = cl.Context(devices=[device])
           break
       i += len(gpu_devices)

    # print('context', ctx)
    q = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    d = np.zeros((1024*1024 * 32 * 2,), dtype=np.float32)
    d_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=d)

    out = np.zeros((1024,), dtype=np.float32)
    out_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=out)

    return ctx, q, mf

def getComputeCapability():
    compute_capability = '%s.%s' % (
        device.get_info(cl.device_info.COMPUTE_CAPABILITY_MAJOR_NV),
        device.get_info(cl.device_info.COMPUTE_CAPABILITY_MINOR_NV)
    )
    return compute_capability

def clearComputeCache():
    cache_dir = join(os.environ['HOME'], '.nv/ComputeCache')
    for subdir in os.listdir(cache_dir):
        if subdir == 'index':
            continue
        # print('clean', subdir)
        subprocess.call(['rm', '-Rf', join(cache_dir, subdir)])
#    subprocess.call(['rm', '-Rf', join(os.environ['HOME'], '.nv/ComputeCache')])

def getPtx(kernelName):
    with open('/tmp/gpucmd.sh', 'w') as f:
        f.write(r"""#!/bin/bash
        cat $(grep -r %s ~/.nv/ComputeCache | awk '{print $3}')
"""  % kernelName)
    ptxb = subprocess.check_output(['/bin/bash', '/tmp/gpucmd.sh'])
    ptx_utf8 = ''
    for byte in ptxb:
        # print(byte)
        if byte >= 10 and byte < 128:
           if chr(byte) in string.printable:
               ptx_utf8 += chr(byte)
    # print('filepath', filepath)
    #print(kernelName)
    ptx = ptx_utf8.split('--opt-level')[0].split('--reserve')[0]
    # print(ptx)
    return ptx

def dumpSass(kernelName):
    ptx = getPtx(kernelName)
    # ptx = ptx.split('.version %s' % getComputeCapability())[1].split('A')[0].split('--reserve-null-pointer')[0]
    ptx = '.target' + ptx.split('.target)[1].split('A')[0].split('--reserve-null-pointer')[0]
    ptx = '.version 4.3\n' + ptx
    # print('ptx', ptx)
    # sys.exit(1)
    with open('/tmp/~kernel.ptx', 'w') as f:
        f.write(ptx)
    print(subprocess.check_output([
        'ptxas',
        '--gpu-name', 'sm_%s' % (getComputeCapability().replace('.', '')),
        '--output-file', '/tmp/~kernel.o',
        '/tmp/~kernel.ptx']).decode('utf-8'))
    sass = subprocess.check_output([
        'cuobjdump', '--dump-sass', '/tmp/~kernel.o']).decode('utf-8')
    print(sass)
    # sys.exit(1)
    return sass

def timeKernel(name, kernel, add_args=[], grid_x=1, block_x=32):
    # clearComputeCache()
    grid = (grid_x,1,1)
    block = (block_x,1,1)
    q.finish()
    inittime()
    call_cl_kernel(kernel, q, grid, block, d_cl, out_cl, *add_args)
    q.finish()
    return timecheck(name)
    # print(getPtx('mykernel'))

def buildKernel(name, source, options=''):
    # options = '-cl-opt-disable'
    #if name in optimized:
    #    print('ENABLING OPTIMIZATIONS')
    #    options = ''
    return cl.Program(ctx, source).build(options=options).__getattr__(name) 

