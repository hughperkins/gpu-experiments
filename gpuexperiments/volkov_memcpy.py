# Note that this will erase your nvidia cache, ~/.nv/ComputeCache  This may or may not be an undesirable side-effect for you.  For example, cutorch will take 1-2 minutes or so to start after this cache has been emptied.

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
# import gpuexperiments.cpu_check
from gpuexperiments.timecheck import inittime, timecheck
import lib_clgpuexp
from lib_clgpuexp import clearComputeCache, getPtx, timeKernel, buildKernel, initClGpu


parser = argparse.ArgumentParser()
parser.add_argument('--printptx', type=bool, default=False)
args = parser.parse_args()

initClGpu()

times = []

# data comes from http://developer.download.nvidia.com/compute/cuda/CUDA_Occupancy_calculator.xls
compute_capability_characteristics = {
    '5.0': {'shared_memory_per_sm': 65536},
    '5.2': {'shared_memory_per_sm': 98304},
    '6.1': {'shared_memory_per_sm': }
}

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

shared_memory_per_sm = compute_capability_characteristics['%s.%s' % compute_capability]['shared_memory_per_sm']

code_template = r"""
    kernel void {{kernelname}} (global {{type}} *data, global {{type}} *out, local float *F) {
        int blockSize = get_local_size(0);
        int iblock = get_group_id(0);
        int index = get_local_id(0) + {{ilp}} * iblock * blockSize;

        {% for i in range(ilp) %}
            {{type}} a{{i}} = data[index + {{i}} * blockSize];
        {% endfor %}

        {% for i in range(ilp) %}
            out[index + {{i}} * blockSize] = a{{i}};
        {% endfor %}
    }
"""

experiments = [
    {'name': 'memcpy_ilp1_float_bsm{bsm}', 'code': code_template, 'ilp': 1, 'type': 'float'},
    {'name': 'memcpy_ilp2_float_bsm{bsm}', 'code': code_template, 'ilp': 2, 'type': 'float'},
    {'name': 'memcpy_ilp4_float_bsm{bsm}', 'code': code_template, 'ilp': 4, 'type': 'float'},
    {'name': 'memcpy_ilp8_float_bsm{bsm}', 'code': code_template, 'ilp': 8, 'type': 'float'},
    {'name': 'memcpy_ilp8_float2_bsm{bsm}', 'code': code_template, 'ilp': 8, 'type': 'float2'},
    {'name': 'memcpy_ilp8_float4_bsm{bsm}', 'code': code_template, 'ilp': 8, 'type': 'float4'}
]

full_occupancy_bsm = 32  # this should probably not be hard coded...
for experiment in experiments:
    # its = (4000000//256//experiment['template_args']['ilp']) * 256 * experiment['template_args']['ilp']
    template = jinja2.Template(experiment['code'], undefined=jinja2.StrictUndefined)
    # for block in range(128,1024+128,128):
    # for occupancy in range(10, 110, 10):
    bsm_done = set()
    typeSize = int((experiment['type'].replace('float', '') + '1')[0]) * 4

    for blocks_per_sm in range(2, 32 + 2, 2):
        block = 32
        grid = 2 * 1024 * 1024
        grid = grid // experiment['ilp']
        grid = grid // (typeSize // 4)
        if experiment['type'] != 'float4':
            if blocks_per_sm == 2:
                grid = grid // 6
            elif blocks_per_sm == 4:
                grid = grid // 2

        shared_bytes = shared_memory_per_sm // blocks_per_sm
        shared_bytes = ((shared_bytes + 0) // 256) * 256
        if shared_bytes >= maxShared * 1024:
            print('exceeds maximum block local memory => skipping')
            continue
        actual_blocks_per_sm = shared_memory_per_sm // shared_bytes
        occupancy = actual_blocks_per_sm / full_occupancy_bsm * 100

        print('occupancy', occupancy, 'shared_bytes', shared_bytes, 'blocks_per_sm', blocks_per_sm,
              'actual_blocks_per_sm', actual_blocks_per_sm, 'shared_memory_per_sm', shared_memory_per_sm)

        if actual_blocks_per_sm in bsm_done:
            continue
        bsm_done.add(actual_blocks_per_sm)
        name = experiment['name'].format(bsm=actual_blocks_per_sm)
        if args.printptx:
            clearComputeCache()
        source = template.render(kernelname=name, **experiment)
        # print('source', source)
        try:
            kernel = buildKernel(name, source)
            print('built kernel')
            for it in range(2):
                t = timeKernel(name, kernel, grid_x=grid, block_x=block, add_args=[
                    cl.LocalMemory(shared_bytes)
                ])
            t_sum = 0
            for it in range(3):
                t_sum += timeKernel(name, kernel, grid_x=grid, block_x=block, add_args=[
                    cl.LocalMemory(shared_bytes)
                ])
            # print(getPtx(name))
            t = t_sum / 3
        except Exception as e:
            print(e)
            break

        # flops = its * block / (t/1000) * 2
        # * 2, because we copy data in both directions, ie twice
        bandwidth_gib = grid * block * experiment['ilp'] * 2 * typeSize / (t / 1000) / 1024 / 1024 / 1024
        print('bandwidth_gib', bandwidth_gib)
        times.append({'name': name, 'time': t, 'bandwidth_gib': bandwidth_gib})


f = open('/tmp/volkov_memcpy_%s.tsv' % deviceSimpleName, 'w')
line = 'name\ttot ms\tbw gib'
print(line)
f.write(line + '\n')
for time_info in times:
    line = '%s\t%.1f\t%.2f' % (time_info['name'], time_info['time'], time_info['bandwidth_gib'])
    print(line)
    f.write(line + '\n')
f.close()
