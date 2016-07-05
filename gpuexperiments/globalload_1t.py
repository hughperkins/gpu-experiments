# Note that this will erase your nvidia cache, ~/.nv/ComputeCache  This may or may not be an undesirable side-effect for you.  For example, cutorch will take 1-2 minutes or so to start after this cache has been emptied.

from __future__ import print_function, division
import time
import string
import random
import numpy as np
import pyopencl as cl
import subprocess
import jinja2
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
    {'name': 'kernel_ilp1t_1', 'source': r"""
kernel void kernel_ilp1t_1(global float *data, global float *out) {
    // lets load some floats...
    int tid = get_local_id(0);
    if(tid != 0) {
        return;
    }
    float sum = 0.0f;
    for(int i = 0; i < (51200); i++) {
        sum += data[i];
    }
    out[0] = sum;
}
"""},
   {'name': 'kernel_ilp1t_2', 'source': r"""
kernel void kernel_ilp1t_2(global float *data, global float *out) {
    // lets load some floats...
    int tid = get_local_id(0);
    if(tid != 0) {
        return;
    }
    float sum = 0.0f;
    for(int i = 0; i < (51200>>5); i++) {
        for(int offset = 0; offset < 51200; offset+=(51200>>5)) {
            sum += data[i + offset];
        }
    }
    out[0] = sum;
}
"""},
   {'name': 'kernel_ilp1t_3', 'source': r"""
kernel void kernel_ilp1t_3(global float *data, global float *out) {
    // lets load some floats...
    int tid = get_local_id(0);
    if(tid != 0) {
        return;
    }
    float sum = 0.0f;
    for(int i = 0; i < (51200>>5); i++) {
        #pragma unroll
        for(int offset = 0; offset < 51200; offset+=(51200>>5)) {
            sum += data[i + offset];
        }
    }
    out[0] = sum;
}
"""},
    {'name': 'kernel_ilp1t_4', 'source': r"""
kernel void kernel_ilp1t_4(global float *data, global float *out) {
    // lets load some floats...
    int tid = get_local_id(0);
    if(tid != 0) {
        return;
    }
    float L[32];
    float sum = 0.0f;
    for(int i = 0; i < (51200); i+=32) {
        #pragma unroll
        for(int j = 0; j < 32; j++) {
            L[j] = data[i + j];
        }
        #pragma unroll
        for(int j = 0; j < 32; j++) {
            sum += L[j];
        }
    }
    out[0] = sum;
}
"""},
    {'name': 'kernel_ilp1t_5', 'source': r"""
kernel void kernel_ilp1t_5(global float *data, global float *out) {
    // lets load some floats...
    int tid = get_local_id(0);
    if(tid != 0) {
        return;
    }
    float L[32];
    float sum = 0.0f;
    for(int i = 0; i < (51200); i+=32) {
        #pragma unroll
        for(int j = 0; j < 32; j++) {
            L[j] = data[i + j];
        }
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        #pragma unroll
        for(int j = 0; j < 32; j+=4) {
            sum0 += L[j + 0];
            sum1 += L[j + 1];
            sum2 += L[j + 2];
            sum3 += L[j + 3];
        }
        sum += (sum0 + sum1) + (sum2 + sum3);
    }
    out[0] = sum;
}
"""}
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
    options = ''
    if name in optimized:
        print('ENABLING OPTIMIZATIONS')
        options = ''
    return cl.Program(ctx, source).build(options=options).__getattr__(name) 

d = np.zeros((1024*1024 * 32 * 2,), dtype=np.float32)
np.random.seed(123)
d[:] = np.random.uniform(size=d.shape)
#print(d.max())
#print(d.min())
#sys.exit(1)
d_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=d)

out = np.zeros((256,), dtype=np.float32)
out_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=out)

def timeKernel(name, kernel):
    # clearComputeCache()
    grid = (1,1,1)  # just one workgroup
    block = (1,1,1)
    q.finish()
    inittime()
    call_cl_kernel(kernel, q, grid, block, d_cl, out_cl)
    q.finish()
    cl.enqueue_copy(q, out, out_cl)
    q.finish()
    print(out[0])
    return timecheck(name), out[0]
    # print(getPtx('mykernel'))

last_sum = None
times = {}
#for name, source in sorted(sources.items()):
for info in sources:
    name = info['name']
    source = info['source']
    clearComputeCache()
    template = jinja2.Template(source, undefined=jinja2.StrictUndefined)
    source = template.render()
    kernel = buildKernel(name, source)
    print('built kernel')
    t = timeKernel(name, kernel)
    t = timeKernel(name, kernel)
    times_sum = 0
    its = 3
    for it in range(its):
        this_time, this_sum = timeKernel(name, kernel)
        if last_sum is None:
            last_sum = this_sum
        else:
            if abs(last_sum - this_sum) > 2:
                print('%s last %s this %s' % (name, last_sum, this_sum))
                assert last_sum == this_sum
        times_sum += this_time
    times[name] = times_sum / its
    print(getPtx(name))
for name, time in sorted(times.items()):
    print(name, time)

