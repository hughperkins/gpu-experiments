# Note that this will erase your nvidia cache, ~/.nv/ComputeCache  This may or may not be an undesirable side-effect for you.  For example, cutorch will take 1-2 minutes or so to start after this cache has been emptied.

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
    kernel void {{name}} (global float *data, global float *out, local float *F) {
        int iblock = get_group_id(0);
        int index = get_local_id(0) + iblock * get_local_size(0);

        float a0 = data[index];
        out[index] = a0;
    }
"""

experiments = [
    {'name': 'memcpy_bsm{bsm}', 'code': code_template, 'ilp': 1}
]

full_occupancy_bsm = 32  # this should probably not be hard coded...
for experiment in experiments:
    # its = (4000000//256//experiment['template_args']['ilp']) * 256 * experiment['template_args']['ilp']
    block = 32
    grid = 1024 * 1024
    template = jinja2.Template(experiment['code'], undefined=jinja2.StrictUndefined)
    # for block in range(128,1024+128,128):
    # for occupancy in range(10, 110, 10):
    bsm_done = set()
    for blocks_per_sm in range(2, 32 + 2, 2):
        shared_bytes = shared_memory_per_sm // blocks_per_sm
        shared_bytes = ((shared_bytes + 0) // 256) * 256
        if shared_bytes >= maxShared * 1024:
            print('exceeds maximum block local memory => skipping')
            continue
        actual_blocks_per_sm = shared_memory_per_sm // shared_bytes
        occupancy = actual_blocks_per_sm / full_occupancy_bsm * 100

        print('occupancy', occupancy,'shared_bytes', shared_bytes, 'blocks_per_sm', blocks_per_sm,
              'actual_blocks_per_sm', actual_blocks_per_sm, 'shared_memory_per_sm', shared_memory_per_sm)

        if actual_blocks_per_sm in bsm_done:
            continue
        bsm_done.add(actual_blocks_per_sm)
        name = experiment['name'].format(bsm=actual_blocks_per_sm)
        clearComputeCache()
        source = template.render(name=name)
        # print('source', source)
        try:
            kernel = buildKernel(name, source)
            print('built kernel')
            for it in range(3):
                t = timeKernel(name, kernel, grid_x=grid, block_x=block, add_args=[
                    cl.LocalMemory(shared_bytes)
                ])
            # print(getPtx(name))
        except Exception as e:
            print(e)
            break

        # flops = its * block / (t/1000) * 2
        bandwidth_gib = grid * block * 4 / (t/1000) / 1024 / 1024 / 1024
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

