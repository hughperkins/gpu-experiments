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

code_template = r"""
    kernel void {{kernelname}} (global float *data, global float *out) {
        int gid = get_global_id(0);

        int blockSize = get_local_size(0);
        int gridSize = get_num_groups(0);
        int outerBlockSize = blockSize * gridSize;

        int offset = 0;
        for(int i = 0; i < {{its}}; i++) {
            out[offset + gid] = 123.0f;
            offset += outerBlockSize;
            offset = offset >= {{maxOffset}} ? 0 : offset;
        }
    }
"""

experiments = [
    {'name': 'gridsize{gridsize}', 'code': code_template}
]

full_occupancy_bsm = 32  # this should probably not be hard coded...
for experiment in experiments:
    if args.exp is not None and args.exp not in experiment['name']:
        continue
    its = 200000
    template = jinja2.Template(experiment['code'], undefined=jinja2.StrictUndefined)
    blockSize = 32
    grid_x = 1
    grid_max = 64 if di.deviceSimpleName == '940m' else 128
    while grid_x <= grid_max:
        name = experiment['name'].format(gridsize=grid_x)
        if args.printptx:
            clearComputeCache()
        maxOffset = 4 * 1024 * 1024
        source = template.render(kernelname=name, its=its, maxOffset=maxOffset, **experiment)
        # print('source', source)
        try:
            kernel = buildKernel(name, source)
            print('built kernel')
            block = (blockSize, 1, 1)
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

        # grid_x *= 2
        if grid_x <= 8:
            grid_x *= 2
        else:
            grid_x += 8


f = open('/tmp/globalwrite_gridsize_%s.tsv' % di.deviceSimpleName, 'w')
line = 'name\ttot ms\tbw gib'
print(line)
f.write(line + '\n')
for time_info in times:
    line = '%s\t%.1f\t%.2f' % (time_info['name'], time_info['time'], time_info['bandwidth_gib'])
    print(line)
    f.write(line + '\n')
f.close()

