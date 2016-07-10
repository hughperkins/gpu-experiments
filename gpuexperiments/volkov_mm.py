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
from lib_clgpuexp import clearComputeCache, getPtx, timeKernel, buildKernel, initClGpu


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

code_template = r"""
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

#define BLOCK_SIZE {{BLOCK_SIZE}}

kernel void {{kernelname}} (global float *C, global float *A, global float *B, int wA, int wB) {
    int bx = get_group_id(0);
    int by = get_group_id(1);

    int tx = get_local_id(0);
    int ty = get_local_id(1);

    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd   = aBegin + wA - 1;
    int aStep  = BLOCK_SIZE;
    int bBegin = BLOCK_SIZE * bx;
    int bStep  = BLOCK_SIZE * wB;

    float Csub = 0;
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        local float As[BLOCK_SIZE][BLOCK_SIZE];
        local float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}
"""

experiments = [
    #{'name': 'memcpy_ilp1_float_bsm{bsm}', 'code': code_template, 'ilp': 1, 'type': 'float'},
]

full_occupancy_bsm = 32  # this should probably not be hard coded...
for experiment in experiments:
    template = jinja2.Template(experiment['code'], undefined=jinja2.StrictUndefined)
    bandwidth_gib = grid * block * experiment['ilp'] * 2 * typeSize / (t/1000) / 1024 / 1024 / 1024
    print('bandwidth_gib', bandwidth_gib)
    times.append({'name': name, 'time': t, 'bandwidth_gib': bandwidth_gib})

f = open('/tmp/volkov_mm_%s.tsv' % deviceSimpleName, 'w')
line = 'name\ttot ms\tbw gib'
print(line)
f.write(line + '\n')
for time_info in times:
    line = '%s\t%.1f\t%.2f' % (time_info['name'], time_info['time'], time_info['bandwidth_gib'])
    print(line)
    f.write(line + '\n')
f.close()

