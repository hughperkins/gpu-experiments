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
from lib_clgpuexp import clearComputeCache, getPtx, timeKernel, buildKernel, initClGpu


initClGpu()

sources = {
    'kernel01': r"""
kernel void kernel01(global float *d, global float *out) {
   float a = 0.0f;
   a += 1.0f;
}
"""
,
    'kernel02': r"""
kernel void kernel02(global float *d, global float *out) {
   float a = 0.0f;
   a += 1.0f;
   a += 1.0f;
   a += 1.0f;
   a += 1.0f;
}
"""
,
    'kernel03': r"""
kernel void kernel03(global float *d, global float *out) {
   float a = 0.0f;
   a += 1.0f;
   a += 1.0f;
   a += 1.0f;
   a += 1.0f;
   
   d[0] = a;
}
"""
,
    'kernel04': r"""
kernel void kernel04(global float *data, global float *out) {
   int a = 1;
   a += 1;
   int b = 1;
   int c = 1;
   int d = 1;
   
   data[0] = a;
   data[0] = b;
   data[0] = c;
   data[0] = d;
}
"""
,
    'kernel05': r"""
kernel void kernel05(global float *data, global float *out) {
   int a = 1;
   a += 1;
   int b = 1;
   int c = 1;
   int d = 1;
   
   //data[0] = a;
   //data[0] = b;
   //data[0] = c;
   //data[0] = d;
}
"""
,
    'kernel06': r"""
kernel void kernel06(global float *data, global float *out) {
   int a = 1;
   a += 1;
   int b = 1;
   int c = 1;
   int d = 1;
   
   data[0] = a;
   //data[0] = b;
   //data[0] = c;
   //data[0] = d;
}
"""
,
    'kernel07': r"""
kernel void kernel07(global float *data, global float *out) {
   int a = 3;
   a += 1;
   int b = 5;
   int c = 6;
   int d = 7;
   
   data[0] = a;
   data[0] = b;
   data[0] = c;
   data[0] = d;
}
"""
,
    'kernel08': r"""
kernel void kernel08(global int *data, global float *out) {
   int a = 3;
   a += 1;
   int b = 5;
   int c = 6;
   int d = 7;
   
   data[0] = a;
   data[0] = b;
   data[0] = c;
   data[0] = d;
}
"""
,
    'kernel09': r"""
kernel void kernel09(global int *data, global int *out) {
   data[get_global_id(0)] = 3;
}
"""
,
    'kernel10': r"""
kernel void kernel10(global int *data, global int *out) {
   data[get_local_id(0)] = 3;
}
"""
,
    'kernel11': r"""
kernel void kernel11(global int *data, global int *out) {
   data[get_global_id(0) << 1] = 3;
}
"""
,
    'kernel12': r"""
kernel void kernel12(global int *data, global int *out) {
   data[get_local_id(0) << 7] = 3;
}
"""
,
    'kernel13': r"""
kernel void kernel13(global int *data, global int *out) {
   data[(get_local_id(0) << 7) + 0] = 3;
   data[(get_local_id(0) << 7) + 1] = 4;
   data[(get_local_id(0) << 7) + 2] = 5;
   data[(get_local_id(0) << 7) + 3] = 6;
}
"""
,
    'kernel14': r"""
kernel void kernel14(global int *data, global int *out) {
   int tid = get_local_id(0);
   data[(tid << 7) + 0] = 3;
   data[(tid << 7) + 1] = 4;
   data[(tid << 7) + 2] = 5;
   data[(tid << 7) + 3] = 6;
}
"""
,
    'kernel15': r"""
kernel void kernel15(global int *data, global int *out) {
   int tid = get_local_id(0);
   int offset = (tid << 7);
   data[offset + 0] = 3;
   data[offset + 1] = 4;
   data[offset + 2] = 5;
   data[offset + 3] = 6;
}
"""
,
    'kernel16': r"""
kernel void kernel16(global int *data, global int *out) {
   int tid = get_local_id(0);
   int offset = tid;
   data[offset + 0] = 3;
   data[offset + 1] = 4;
   data[offset + 2] = 5;
   data[offset + 3] = 6;
}
"""
,
    'kernel17': r"""
kernel void kernel17(global int *data, global int *out) {
    local int foo[32];
    foo[0] = 31;
}
"""
,
    'kernel18': r"""
kernel void kernel18(global int *data, global int *out) {
    int a = 31;
    a += 1;
    local int foo[32];
    foo[0] = a;
}
"""
,
    'kernel19': r"""
kernel void kernel19(global int *data, global int *out) {
    int a = 31;
    int b = 5;
    a += b;
    local int foo[32];
    foo[0] = a;
}
"""
,
    'kernel20': r"""
kernel void kernel20(global int *data, global int *out) {
    int a = 31;
    for(int i = 0; i < 4; i++) {
        a += 1;
    }
    int b = 5;
    a += b;
    local int foo[32];
    foo[0] = a;
}
"""
,
    'kernel21': r"""
kernel void kernel21(global int *data, global int *out) {
    int gid = get_global_id(0);
    data[0] = gid;
}
"""
,
    'kernel22': r"""
kernel void kernel22(global int *data, global int *out) {
    int gid = get_global_id(0);
    data[gid] = gid;
}
"""
,
    'kernel23': r"""
kernel void kernel23(global int *data, global int *out) {
    int tid = get_local_id(0);
    data[0] = tid;
}
"""
,
    'kernel24': r"""
kernel void kernel24(global int *data, global int *out) {
    int tid = get_local_id(0);
    data[tid] = tid;
}
"""
,
    'kernel25': r"""
kernel void kernel25(global int *data, global int *out) {
    data[0] = 5;
}
"""
,
    'kernel26': r"""
kernel void kernel26(global int *data, global int *out) {
    int tid = get_local_id(0);
    data[0] = tid;
    data[0] = tid;
}
"""
,
    'kernel27': r"""
kernel void kernel27(global int *data, global int *out) {
    int tid = get_local_id(0);
    data[0] = tid;
    data[0] = get_local_id(0);
}
"""
,
    'kernel28': r"""
kernel void kernel28(global int *data, global int *out) {
    int gid = get_global_id(0);
    data[gid % 1024 * 1024] = gid;
}
"""
,
    'kernel29': r"""
kernel void kernel29(global int *data, global int *out) {
    int gid = get_global_id(0);
    data[(gid << 1) % 1024 * 1024] = gid;
}
"""
,
    'kernel30': r"""
kernel void kernel30(global int *data, global int *out) {
    int gid = get_global_id(0);
    data[(gid << 2) % 1024 * 1024] = gid;
}
"""
,
    'kernel31': r"""
kernel void kernel31(global int *data, global int *out) {
    int gid = get_global_id(0);
    data[(gid << 3) % 1024 * 1024] = gid;
}
"""
,
    'kernel32': r"""
kernel void kernel32(global int *data, global int *out) {
    int gid = get_global_id(0);
    data[(gid << 4) % (1024 * 1024)] = gid;
}
"""
}

#sources = {'kernel1': kernel1, 'kernel2': kernel2, 'kernel3': kernel3}
#kernels = {}
#for name, source in ({'1': kernel1, '2': kernel2, '3': kernel3}).items():
#    # -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros
#    kernels[name] = cl.Program(ctx, source).build(options='-cl-opt-disable').__getattr__('mykernel') 


times = []
for name, source in sorted(sources.items()):
    clearComputeCache()
    kernel = buildKernel(name, source, options='-cl-opt-disable')
    print('built kernel')
    for it in range(3):
        t = timeKernel(name, kernel, grid_x=1024*1024, block_x=32)
        # times[name] = t
    times.append({'name': name, 'time': t})
    print(getPtx(name))

print('name\t\ttot ms')
for timeinfo in times:
    print('%s\t%.1f' % (timeinfo['name'], timeinfo['time']))

