from __future__ import print_function, division
import argparse
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
from gpuexperiments.timecheck import inittime, timecheck
import lib_clgpuexp
from lib_clgpuexp import clearComputeCache, getPtx, timeKernel3d, buildKernel, initClGpu
from gpuexperiments.deviceinfo import DeviceInfo


parser = argparse.ArgumentParser()
parser.add_argument('--printptx', type=bool, default=False, help='note that it erases your nv cache')
parser.add_argument('--exp')
args = parser.parse_args()

initClGpu()

times = []

di = DeviceInfo(lib_clgpuexp.device)

"""
Layout for this is

<-   memory block size   -><-   memory block size   -><-   memory block size   ->
t1 t2 t3                   [blank, skipped]           t1 t2 t3

"""
code_template = r"""
    kernel void {{kernelname}} (global float *data, global float *out) {
        int memBlockSize = {{memBlockSize}};
        int tid = get_local_id(0);

        int ourBlockId = tid / memBlockSize;  // we only do this once, so fine to be slow...
        int ourIntraBlockId = tid - (ourBlockId * memBlockSize);
        int ourOffset = ourBlockId * 2 * memBlockSize + ourIntraBlockId;
        int threadingMemBlockSize = 64;
        int offset = 0;
        for(int i = 0; i < {{its}}; i++) {
            out[offset + ourOffset] = 123.0f;

            offset += threadingMemBlockSize;
            offset = offset > {{maxOffset}} ? 0 : offset;
        }
    }
"""

experiments = [
    {'name': 'memcpy_memblocksize{memblocksize}', 'code': code_template}
]

full_occupancy_bsm = 32  # this should probably not be hard coded...
for experiment in experiments:
    if args.exp is not None and args.exp not in experiment['name']:
        continue
    its = 10000000
    template = jinja2.Template(experiment['code'], undefined=jinja2.StrictUndefined)
    memblocksize = 1
    while memblocksize <= 32:
        block_x = 32
        grid_x = 1
        name = experiment['name'].format(memblocksize=memblocksize)
        if args.printptx:
            clearComputeCache()
        maxOffset = lib_clgpuexp.out.size // 4 - 64
        source = template.render(kernelname=name, **experiment, its=its, memBlockSize=memblocksize,
                                 maxOffset=maxOffset)
        # print('source', source)
        try:
            kernel = buildKernel(name, source)
            print('built kernel')
            block = (block_x, 1, 1)
            grid = (grid_x, 1, 1)
            for it in range(2):
                t = timeKernel3d(name, kernel, grid=grid, block=block, add_args=[
                ])
            t_sum = 0
            for it in range(3):
                t_sum += timeKernel3d(name, kernel, grid=grid, block=block, add_args=[
                ])
            # print(getPtx(name))
            t = t_sum / 3
        except Exception as e:
            print(e)
            break

        # flops = its * block / (t/1000) * 2
        # * 2, because we copy data in both directions, ie twice
        typeSize = 4
        bandwidth_gib = its * grid[0] * grid[1] * block[0] * block[1] * typeSize / (t/1000) / 1024 / 1024 / 1024
        print('bandwidth_gib', bandwidth_gib)
        times.append({'name': name, 'time': t, 'bandwidth_gib': bandwidth_gib})

        memblocksize *= 2


f = open('/tmp/globalwrite_blocked_%s.tsv' % di.deviceSimpleName, 'w')
line = 'name\ttot ms\tbw gib'
print(line)
f.write(line + '\n')
for time_info in times:
    line = '%s\t%.1f\t%.2f' % (time_info['name'], time_info['time'], time_info['bandwidth_gib'])
    print(line)
    f.write(line + '\n')
f.close()

