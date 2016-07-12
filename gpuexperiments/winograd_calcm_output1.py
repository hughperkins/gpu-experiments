"""
Start playing wtih some of the parts of
https://github.com/hughperkins/neonCl-underconstruction/blob/master/winograd_kernels_cl.py
"""
from __future__ import print_function, division
import time
import string
import random
import jinja2
import argparse
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


parser = argparse.ArgumentParser()
parser.add_argument('--printptx', type=bool, default=False)
args = parser.parse_args()

initClGpu()

times = []

compute_units = lib_clgpuexp.device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
maxShared = lib_clgpuexp.device.get_info(cl.device_info.LOCAL_MEM_SIZE) // 1024
compute_capability = (
    lib_clgpuexp.device.get_info(cl.device_info.COMPUTE_CAPABILITY_MAJOR_NV),
    lib_clgpuexp.device.get_info(cl.device_info.COMPUTE_CAPABILITY_MINOR_NV)
)
deviceName = lib_clgpuexp.device.get_info(cl.device_info.NAME)
deviceSimpleName = deviceName.replace('GeForce', '').replace('GTX', '').strip().replace(' ', '').lower()

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
kernel void calcm_output(
        global float *data, global float * M,
        int tiles, int GN, int GK
    ) {

    int tid1 = get_local_id(1);
    int tid = get_local_id(0);
    int b = get_group_id(0);

    int tiles2 = tiles * tiles;
    int b36 = 36 * b;
    for(int gn = 0; gn < GN; gn++) {
        int gn32 = gn << 5;
        for(int gk = 0; gk < GK; gk++) {
            int gk32 = gk << 5;
           int offset = (gn32) * GK * 32 * tiles2 * 6 * 6 +
                        (gk32) * tiles2 * 6 * 6 +
                        b36
                        ;
            int n_stride = GK * 32 * tiles2 * 6 * 6;
            int co_stride = tiles2 * 6 * 6;
            offset += tid1 * n_stride + tid * co_stride;
            float sum0 = 0.0f;
            for(int xinu = 0; xinu < 36; xinu++) {
                M[offset + xinu] = sum0;
            }
        }
    }
}
"""

blocksize = 32
K = 32
GK = 1
N = 32
batchsize = 32
GN = 1
H = 56
W = 56
tiles = H // 4
S = 32


experiments = [
    {'name': 'calcm_output', 'code': code_template, 'block': (blocksize, blocksize, 1), 'outs': 1}
]

times = []
full_occupancy_bsm = 32  # this should probably not be hard coded...
if args.printptx:
    clearComputeCache()
for experiment in experiments:
    #batchsize = 1
    #while batchsize <= 1024:
    name = experiment['name'].format(batchsize=batchsize)
    template = jinja2.Template(experiment['code'], undefined=jinja2.StrictUndefined)
    source = template.render(kernelname=name, BLOCK_SIZE=blocksize, **experiment)
    # print('source', source)
    kernel = buildKernel(name, source)

    grid = (tiles * tiles, 1, 1)
    block = experiment['block']

    for it in range(2):
        t = timeKernel3d(name, kernel, grid=grid, block=block, add_args=[
            tiles, GN, GK
        ])

    t_sum = 0
    its = 1
    # its *= (1024 // batchsize)
    for it in range(its):
        t_sum += timeKernel3d(name, kernel, grid=grid, block=block, add_args=[
            tiles, GN, GK
        ])
    t = t_sum / its

    bw_gib = grid[0] * block[0] * block[1]
    bw_gib = 0

    # ops = S * S * S * 2
    #ops = S * S * S * 2 * batchsize
    #gflops = ops / (t/1000) / 1000 / 1000 / 1000
    # gflops = -1

    # print(getPtx(name))
    # dumpSass(name)

    times.append({'name': name, 'time': t, 'bw gib': bw_gib})


f = open('/tmp/winograd_calcm_output_%s.tsv' % deviceSimpleName, 'w')
print('')
line = 'name\ttime\tflops'
print(line)
f.write(line + '\n')
for timeinfo in times:
    line = '%s\t%.1f\t%.1f' % (timeinfo['name'], timeinfo['time'], timeinfo['bw gib'])
    print(line)
    f.write(line + '\n')
f.close()

