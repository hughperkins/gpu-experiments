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
Layout for this is, for one-block

<-   blocksize   ->
t1 t2 t3           t1 t2 t3                  t1 t2 t3

"""
code_template = r"""
    kernel void {{kernelname}} (global {{type}} *data, global {{type}} *out, local float *F) {
        int blockSize = get_local_size(0) * {{stride}};
        int iblock = get_group_id(0);
        int stride_idx = get_group_id(1);
        int index = get_local_id(0) * {{stride}} + {{ilp}} * iblock * blockSize + stride_idx;

        {% for i in range(ilp) %}
            out[index + {{i}} * blockSize] = 123.0f;
        {% endfor %}
    }
"""

"""
Layout for this is, for one-block

<-ilp ->
t1 t1 t1 t2 t2 t2 t3 t3 t3

"""
code_template_layout2 = r"""
    kernel void {{kernelname}} (global {{type}} *data, global {{type}} *out, local float *F) {
        int blockSize = get_local_size(0) * {{stride}};
        int iblock = get_group_id(0);
        int index = iblock * blockSize * {{ilp}} + get_local_id(0) * {{ilp}} * {{stride}};

        {% for i in range(ilp) %}
            out[index + {{i}} * {{stride}}] = 123.0f;
        {% endfor %}
    }
"""

experiments = [
    {'name': 'memcpy_ilp1_float_bsm{bsm}', 'code': code_template, 'ilp': 1, 'type': 'float'},
    {'name': 'memcpy_ilp2_float_bsm{bsm}', 'code': code_template, 'ilp': 2, 'type': 'float'},
    {'name': 'memcpy_ilp4_float_bsm{bsm}', 'code': code_template, 'ilp': 4, 'type': 'float'},
    {'name': 'memcpy_ilp8_float_bsm{bsm}', 'code': code_template, 'ilp': 8, 'type': 'float'},
    {'name': 'memcpy_ilp8_float2_bsm{bsm}', 'code': code_template, 'ilp': 8, 'type': 'float2'},
    {'name': 'memcpy_ilp8_float4_bsm{bsm}', 'code': code_template, 'ilp': 8, 'type': 'float4'},
    {'name': 'memcpy_l2_ilp1_float_bsm{bsm}', 'code': code_template_layout2, 'ilp': 1, 'type': 'float'},
    {'name': 'memcpy_l1_stride2_ilp1_float_bsm{bsm}', 'code': code_template_layout2, 'ilp': 1, 'type': 'float', 'stride': 2},
    {'name': 'memcpy_l1_stride3_ilp1_float_bsm{bsm}', 'code': code_template_layout2, 'ilp': 1, 'type': 'float', 'stride': 3},
    {'name': 'memcpy_l1_stride4_ilp1_float_bsm{bsm}', 'code': code_template_layout2, 'ilp': 1, 'type': 'float', 'stride': 4},
    {'name': 'memcpy_l1_stride8_ilp1_float_bsm{bsm}', 'code': code_template_layout2, 'ilp': 1, 'type': 'float', 'stride': 8}
    #{'name': 'memcpy_l2_stride2_ilp1_float_bsm{bsm}', 'code': code_template_layout2, 'ilp': 1, 'type': 'float', 'stride': 2},
]

full_occupancy_bsm = 32  # this should probably not be hard coded...
for experiment in experiments:
    if args.exp is not None and args.exp not in experiment['name']:
        continue
    # its = (4000000//256//experiment['template_args']['ilp']) * 256 * experiment['template_args']['ilp']
    template = jinja2.Template(experiment['code'], undefined=jinja2.StrictUndefined)
    # for block in range(128,1024+128,128):
    # for occupancy in range(10, 110, 10):
    bsm_done = set()
    typeSize = int((experiment['type'].replace('float', '') + '1')[0]) * 4

    for blocks_per_sm in range(2, 16 + 2, 2):
        block = 32
        grid_x = 2 * 1024 * 1024
        grid_x = grid_x // experiment['ilp']
        grid_x = grid_x // (typeSize//4)
        if experiment['type'] != 'float4':
            if blocks_per_sm == 2:
                grid_x = grid_x // 6
            elif blocks_per_sm == 4:
                grid_x = grid_x // 2

        shared_bytes = di.shared_memory_per_sm // blocks_per_sm
        shared_bytes = ((shared_bytes + 0) // 256) * 256
        if shared_bytes >= di.maxShared * 1024:
            print('exceeds maximum block local memory => skipping')
            continue
        actual_blocks_per_sm = di.shared_memory_per_sm // shared_bytes
        occupancy = actual_blocks_per_sm / full_occupancy_bsm * 100

        experiment['stride'] = experiment.get('stride', 1)
        print('occupancy', occupancy,'shared_bytes', shared_bytes, 'blocks_per_sm', blocks_per_sm,
              'actual_blocks_per_sm', actual_blocks_per_sm, 'shared_memory_per_sm', di.shared_memory_per_sm)

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
            grid = (grid_x // experiment['stride'], experiment['stride'], 1)
            for it in range(2):
                t = timeKernel3d(name, kernel, grid=grid, block=(block, 1, 1), add_args=[
                    cl.LocalMemory(shared_bytes)
                ])
            t_sum = 0
            for it in range(3):
                t_sum += timeKernel3d(name, kernel, grid=grid, block=(block, 1, 1), add_args=[
                    cl.LocalMemory(shared_bytes)
                ])
            # print(getPtx(name))
            t = t_sum / 3
        except Exception as e:
            print(e)
            break

        # flops = its * block / (t/1000) * 2
        # * 2, because we copy data in both directions, ie twice
        bandwidth_gib = grid[0] * grid[1] * block * experiment['ilp'] * typeSize / (t/1000) / 1024 / 1024 / 1024
        print('bandwidth_gib', bandwidth_gib)
        times.append({'name': name, 'time': t, 'bandwidth_gib': bandwidth_gib})


f = open('/tmp/globalwrite_%s.tsv' % di.deviceSimpleName, 'w')
line = 'name\ttot ms\tbw gib'
print(line)
f.write(line + '\n')
for time_info in times:
    line = '%s\t%.1f\t%.2f' % (time_info['name'], time_info['time'], time_info['bandwidth_gib'])
    print(line)
    f.write(line + '\n')
f.close()

