# Note that this will erase your nvidia cache, ~/.nv/ComputeCache  This may or may not be an undesirable side-effect for you.  For example, cutorch will take 1-2 minutes or so to start after this cache has been emptied.

"""
This software contains source code provided by NVIDIA Corporation.
"""

from __future__ import print_function, division
import time
import string
import random
import jinja2
import numpy as np
import pyopencl as cl
import subprocess
import os
from os.path import join
from gpuexperiments.callkernel import call_cl_kernel
#import gpuexperiments.cpu_check
from gpuexperiments.timecheck import inittime, timecheck
import lib_clgpuexp
from lib_clgpuexp import clearComputeCache, getPtx, timeKernel3d, buildKernel, initClGpu
from lib_clgpuexp import dumpSass


initClGpu()

times = []

compute_units = lib_clgpuexp.device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
maxShared = lib_clgpuexp.device.get_info(cl.device_info.LOCAL_MEM_SIZE) // 1024
compute_capability = (
    lib_clgpuexp.device.get_info(cl.device_info.COMPUTE_CAPABILITY_MAJOR_NV),
    lib_clgpuexp.device.get_info(cl.device_info.COMPUTE_CAPABILITY_MINOR_NV)
)
deviceName = lib_clgpuexp.device.get_info(cl.device_info.NAME)
deviceSimpleName = deviceName.replace('GeForce', '').strip().replace(' ', '').lower()

print('deviceName', deviceName, 'compute capability', compute_capability)
print('compute units', compute_units, 'max shared memory', maxShared)

shared_memory_per_sm = None
# data comes from http://developer.download.nvidia.com/compute/cuda/CUDA_Occupancy_calculator.xls
if compute_capability[0] == 5:
    if compute_capability[1] == 0:
        shared_memory_per_sm = 65536
    elif compute_capability[1] == 2:
        shared_memory_per_sm = 98304
    else:
        raise Exception('compute capability %s not recognized' % compute_capability)
else:
    raise Exception('compute capability %s not recognized' % compute_capability)
assert shared_memory_per_sm is not None

header = r"""
// adapted from CUDA 7.5 SDK
// original copyright header:
/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Per my understanding, this is licensed under clause 2.1.1 of the EULA:
// 2.1.1. Source Code
// 
// Developer shall have the right to modify and create derivative
// works with the Source Code. Developer shall own any derivative
// works ("Derivatives") it creates to the Source Code, provided
// that Developer uses the Materials in accordance with the terms
// and conditions of this Agreement. Developer may distribute the
// Derivatives, provided that all NVIDIA copyright notices and
// trademarks are propagated and used properly and the
// Derivatives include the following statement: "This software
// contains source code provided by NVIDIA Corporation."
"""

code_template_8 = header + r"""
#define BLOCK_SIZE {{BLOCK_SIZE}}

kernel void {{kernelname}} (global float *A, global float *B, global float *C, int wA, int wB) {
    int bx = get_group_id(0);
    int by = get_group_id(1);

    int tx = get_local_id(0);
    int ty = get_local_id(1);

    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd   = aBegin + wA - 1;
    int aStep  = BLOCK_SIZE;
    int bBegin = BLOCK_SIZE * bx;
    int bStep  = BLOCK_SIZE * wB;

    float Csub[{{outs}}];
    {% for i in range(outs) %}
        Csub[{{i}}] = 0;
    {% endfor %}
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        local float As[BLOCK_SIZE][BLOCK_SIZE];
        local float Bs[BLOCK_SIZE][BLOCK_SIZE];

        {% for i in range(outs) %}
            As[ty + {{i}} * BLOCK_SIZE / {{outs}}][tx] = A[a + wA * (ty + {{i}} * BLOCK_SIZE / {{outs}}) + tx];
            Bs[ty + {{i}} * BLOCK_SIZE / {{outs}}][tx] = B[b + wB * (ty + {{i}} * BLOCK_SIZE / {{outs}}) + tx];
        {% endfor %}

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll 32
        for(int k = 0; k < BLOCK_SIZE; ++k) {
            {% for i in range(outs) %}
                Csub[{{i}}] = fma(As[ty + {{i}} * BLOCK_SIZE / {{outs}}][k], Bs[k][tx], Csub[{{i}}]);
            {% endfor %}
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    {% for i in range(outs) %}
        C[c + wB * (ty + {{i}} * BLOCK_SIZE / {{outs}}) + tx] = Csub[{{i}}];
    {% endfor %}
}
"""

# S = 1024
blocksize=32

experiments = [
    {'name': 'mm1_s{S}', 'code': code_template_8, 'block': (blocksize, blocksize, 1), 'outs': 1},
    {'name': 'mm2_s{S}', 'code': code_template_8, 'block': (blocksize, blocksize//2, 1), 'outs': 2},
    {'name': 'mm4_s{S}', 'code': code_template_8, 'block': (blocksize, blocksize//4, 1), 'outs': 4},
    {'name': 'mm8_s{S}', 'code': code_template_8, 'block': (blocksize, blocksize//8, 1), 'outs': 8}
]

cl = lib_clgpuexp.cl
ctx = lib_clgpuexp.ctx
q = lib_clgpuexp.q
mf = lib_clgpuexp.mf

times = []
full_occupancy_bsm = 32  # this should probably not be hard coded...
clearComputeCache()
for experiment in experiments:
    S = 32
    while S <= 1024:
        name = experiment['name'].format(S=S)
        template = jinja2.Template(experiment['code'], undefined=jinja2.StrictUndefined)
        source = template.render(kernelname=name, BLOCK_SIZE=blocksize, **experiment)
        # print('source', source)
        kernel = buildKernel(name, source)

        grid = (S//blocksize, S//blocksize, 1)
        block = experiment['block']

        #A = np.zeros((S,S), dtype=np.float32)
        #B = np.zeros((S,S), dtype=np.float32)
        lib_clgpuexp.d = np.zeros((S,S), dtype=np.float32)
        d = lib_clgpuexp.d
        d[:] = np.random.rand(d.size).reshape(S,S) - 0.5
        lib_clgpuexp.d_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=d)
        A = d.reshape(S,S)

        lib_clgpuexp.out = np.zeros((S,S), dtype=np.float32)
        out = lib_clgpuexp.out
        out[:] = np.random.rand(out.size).reshape(S,S) - 0.5
        lib_clgpuexp.out_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=out)
        B = out.reshape(S,S)

        C = np.zeros((S,S), dtype=np.float32)
        C_cl = cl.Buffer(lib_clgpuexp.ctx, lib_clgpuexp.mf.READ_WRITE | lib_clgpuexp.mf.COPY_HOST_PTR, hostbuf=C)

        for it in range(2):
            t = timeKernel3d(name, kernel, grid=grid, block=block, add_args=[
                C_cl, S, S
            ])

        t_sum = 0
        for it in range(5):
            t_sum += timeKernel3d(name, kernel, grid=grid, block=block, add_args=[
                C_cl, S, S
            ])
        t = t_sum / 5

        ops = S * S * S * 2
        gflops = ops / (t/1000) / 1000 / 1000 / 1000

        cl.enqueue_copy(q, C, C_cl)
        q.finish()
        print(C[0,:10])
        print(A.dot(B)[0,:10])

        C_cpu = A.dot(B)
        cpu_samples = ''
        gpu_samples = ''
        diffs = ''
        for sample in range(20):
            x = random.randint(0, S - 1)
            y = random.randint(0, S - 1)
            c_gpu = C[x,y]
            c_local = C_cpu[x,y]
            diff = abs(c_gpu - c_local)
            cpu_samples += ' %.4f' % c_local
            gpu_samples += ' %.4f' % c_gpu
            diffs += ' %.5f' % diff
            # print(c_gpu, c_local, diff)
            assert diff < 1e-4
        print('cpu', cpu_samples)
        print('gpu', gpu_samples)
        print('diffs', diffs)

        # print(getPtx(name))
        # dumpSass(name)

        times.append({'name': name, 'time': t, 'gflops': gflops})
        print('name', name, 't', t, 'gflops', gflops)

        if S <= 128:
            S *= 2
        else:
            S += 128

f = open('/tmp/volkov_mm_%s.tsv' % deviceSimpleName, 'w')
print('')
line = 'name\ttime\tflops'
print(line)
f.write(line + '\n')
for timeinfo in times:
    line = '%s\t%.1f\t%.1f' % (timeinfo['name'], timeinfo['time'], timeinfo['gflops'])
    print(line)
    f.write(line + '\n')
f.close()

