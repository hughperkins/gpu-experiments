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

code_reduce = r"""
    local sums[32];
    sums[tid] = sum;
    // barrier(CLK_LOCAL_MEM_FENCE); // cos we are warp, so no need...
    if(tid == 0) {
        sum = 0.0;
        for(int i = 0; i < 32; i++) {
            sum += sums[i];
        }
        out[0] = sum;
    }
"""

sources = {
    'kernel_ilp_load1': r"""
kernel void kernel_ilp_load1(global float *data, global float *out) {
    // lets load some floats...
    int tid = get_local_id(0);
    int offset = tid * 51200;
    float sum = 0.0f;
    for(int i = 0; i < 51200; i++) {
        sum += data[i + offset];
    }
    {{reduce}}
}
"""
,    'kernel_ilp_load2': r"""
kernel void kernel_ilp_load2(global float *data, global float *out) {
    // lets load some floats...
    int tid = get_local_id(0);
    float sum = 0.0f;
    for(int i = 0; i < 51200; i++) {
        sum += data[(i<<5) + tid];
    }
    {{reduce}}
}
"""
,    'kernel_ilp_load2b': r"""
kernel void kernel_ilp_load2b(global float *data, global float *out) {
    // lets load some floats...
    int tid = get_local_id(0);
    float sum = 0.0f;
    for(int i = 0; i < (51200<<5); i+=32) {
        sum += data[i + tid];
    }
    {{reduce}}
}
"""
,
    'kernel_ilp_load3': r"""
kernel void kernel_ilp_load3(global float *data, global float *out) {
    // lets load some floats...
    int tid = get_local_id(0);
    float sum = 0.0f;
    for(int i = 0; i < 25600; i++) {
        sum += data[(i<<6) + tid];
        sum += data[(i<<6) + tid + 32];
    }
    {{reduce}}
}
"""
,
    'kernel_ilp_load4': r"""
kernel void kernel_ilp_load4(global float *data, global float *out) {
    int tid = get_local_id(0);
    float sum = 0.0f;
    for(int i = 0; i < 12800; i++) {
        sum += data[(i<<7) + tid];
        sum += data[(i<<7) + tid + 32];
        sum += data[(i<<7) + tid + 64];
        sum += data[(i<<7) + tid + 96];
    }
    {{reduce}}
}
"""
,
    'kernel_ilp_load5': r"""
kernel void kernel_ilp_load5(global float *data, global float *out) {
    int tid = get_local_id(0);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for(int i = 0; i < 12800; i++) {
        sum0 += data[(i<<7) + tid];
        sum1 += data[(i<<7) + tid + 32];
        sum2 += data[(i<<7) + tid + 64];
        sum3 += data[(i<<7) + tid + 96];
    }
    float sum = (sum0 + sum1) + (sum2 + sum3);
    {{reduce}}
}
"""
,
    'kernel_ilp_load1b': r"""
kernel void kernel_ilp_load1b(global float *data, global float *out) {
    // lets load some floats...
    int tid = get_local_id(0);
    int offset = tid * 51200;
    float sum = 0.0f;
    for(int i = 0; i < 51200; i+= (1<<3)) {
        for( int j = 0; j < (1<<3); j++) {
            sum += data[i + j + offset];
        }
    }
    {{reduce}}
}
"""
,
    'kernel_ilp_load1c': r"""
kernel void kernel_ilp_load1c(global float *data, global float *out) {
    // lets load some floats...
    int tid = get_local_id(0);
    int offset = tid * 51200;
    float sum = 0.0f;
    for(int i = 0; i < 51200; i+= (1<<3)) {
        #pragma unroll
        for( int j = 0; j < (1<<3); j++) {
            sum += data[i + j + offset];
        }
    }
    {{reduce}}
}
"""
,
    'kernel_ilp_load1d': r"""
kernel void kernel_ilp_load1d(global float *data, global float *out) {
    // lets load some floats...
    int tid = get_local_id(0);
    int offset = tid * 51200;
    float sum_[4];
    sum_[0] = 0;
    sum_[1] = 0;
    sum_[2] = 0;
    sum_[3] = 0;
    for(int i = 0; i < 51200; i+= (1<<2)) {
        sum_[0] += data[i + 0 + offset];
        sum_[1] += data[i + 1 + offset];
        sum_[2] += data[i + 2 + offset];
        sum_[3] += data[i + 3 + offset];
    }
    float sum = (sum_[0] + sum_[1]) + (sum_[2] + sum_[3]);
    {{reduce}}
}
"""
,
    'kernel_ilp_load1e': r"""
kernel void kernel_ilp_load1e(global float *data, global float *out) {
    // lets load some floats...
    int tid = get_local_id(0);
    int offset = tid * 51200;
    float sum_[4];
    sum_[0] = 0;
    sum_[1] = 0;
    sum_[2] = 0;
    sum_[3] = 0;
    int offset4 = offset >> 2;
    global float4 *data4 = (global float4 *)data;
    float4 val4[1];
    float *val = (float *)val4;
    for(int i = 0; i < (51200 >> 2); i++) {
        val4[0] = data4[i + offset4];
        sum_[0] += val[0];
        sum_[1] += val[1];
        sum_[2] += val[2];
        sum_[3] += val[3];
    }
    float sum = (sum_[0] + sum_[1]) + (sum_[2] + sum_[3]);
    {{reduce}}
}
"""
,
    'kernel_ilp_load1f': r"""
kernel void kernel_ilp_load1f(global float *data, global float *out) {
    // lets load some floats...
    int tid = get_local_id(0);
    int offset = tid * 51200;
    float sum_[8];
    {% for i in range(8) %}
      sum_[{{i}}] = 0;
    {% endfor %}
    int offset4 = offset >> 2;
    global float4 *data4 = (global float4 *)data;
    float4 val4[2];
    float *val = (float *)val4;
    for(int i = 0; i < (51200 >> 2); i+= 2) {
        val4[0] = data4[i + offset4];
        val4[1] = data4[i + 1 + offset4];
        sum_[0] += val[0];
        sum_[1] += val[1];
        sum_[2] += val[2];
        sum_[3] += val[3];

        sum_[4] += val[4];
        sum_[5] += val[5];
        sum_[6] += val[6];
        sum_[7] += val[7];
    }
    float sum = (sum_[0] + sum_[1]) + (sum_[2] + sum_[3]) + sum_[4] + sum_[5] + sum_[6] + sum_[7];
    {{reduce}}
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
    block = (32,1,1)
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
for name, source in sorted(sources.items()):
    clearComputeCache()
    template = jinja2.Template(source, undefined=jinja2.StrictUndefined)
    source = template.render(reduce=code_reduce)
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

