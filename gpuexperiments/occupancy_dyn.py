"""
Try using dynamic shared memory, see if gets optimized away, or affects occupancy
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
from lib_clgpuexp import dumpSass


times = []

code_template = r"""
            kernel void {{name}} (global float *data, global float *out{% if shared %}, local float *F{% endif %}) {
                {% for j in range(ilp) %}
                  float a{{j}} = data[{{j}}];
                {% endfor %}
                float b = data[0];
                float c = data[1];
                #pragma unroll 256
                for(int i = 0; i < {{its / ilp}}; i++) {
                    {% for j in range(ilp) %}
                      {% if fma %}
                        a{{j}} = fma(a{{j}}, b, c);
                      {% else %}
                        a{{j}} = a{{j}} * b + c;
                      {% endif %}
                    {% endfor %}
                }
                float a = 0.0f;
                {% for j in range(ilp) %}
                  a += a{{j}};
                {% endfor %}
                out[0] = a;
            }
        """

experiments = [
    {'name': 'k1_g{grid}_b{block}_s{shared}', 'code': code_template, 'options': '', 'template_args': {'fma': True, 'ilp': 1}, 'block': 32, 'grid': 1024},
    {'name': 'k1_g{grid}_b{block}_s{shared}', 'code': code_template, 'options': '', 'template_args': {'fma': True, 'ilp': 1}, 'block': 64, 'grid': 1024}
]

initClGpu()

compute_units = lib_clgpuexp.device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
maxShared = lib_clgpuexp.device.get_info(cl.device_info.LOCAL_MEM_SIZE) // 1024
compute_capability = (
    lib_clgpuexp.device.get_info(cl.device_info.COMPUTE_CAPABILITY_MAJOR_NV),
    lib_clgpuexp.device.get_info(cl.device_info.COMPUTE_CAPABILITY_MINOR_NV)
)
print('compute capability', compute_capability)
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
for experiment in experiments:

    template = jinja2.Template(experiment['code'], undefined=jinja2.StrictUndefined)
    #for block in range(128,1024+128,128):
    block = experiment['block']
    grid = experiment['grid'] # 1024
    ilp = experiment['template_args']['ilp']
    for shared in range(0, maxShared, 4):
        if shared == 0:
            occupancy = 32
        else:
            occupancy = 64 // shared
        its = 400000000 // grid * 32 // block
        if grid == 1:  # modify its, since it will only run on one sm
            its = its // compute_units
        its = its * occupancy // 32
        its = (its // 256 // ilp) * 256 * ilp
        name = experiment['name'].format(shared=shared, grid=grid, block=block)
        clearComputeCache()
        add_args = []
        if shared > 0:
            add_args.append(cl.LocalMemory(shared*1024))
        source = template.render(name=name, its=its, type='float', shared=shared > 0, **experiment['template_args'])
        # print('source', source)
        try:
            kernel = buildKernel(name, source, options=experiment['options'])
            for it in range(3):
                t = timeKernel(name, kernel, grid_x=grid, block_x=block, add_args=add_args)
            print(getPtx(name))
        except Exception as e:
            print(e)
            break

        flops = its * block / (t/1000) * 2 * grid
        times.append({'name': name, 'time': t, 'flops': flops})
    dumpSass(name)

# try varying occupancy, rather than varying shared memory
# assume shared memory per sm = 65536 bytes (as per sm5.0)
# assume full occupancy is 16 blocks per sm, but I'm not sure why...
full_occupancy_bsm = 32
for blocks_per_sm in range(2, full_occupancy_bsm + 2, 2):
    shared_bytes = shared_memory_per_sm // blocks_per_sm
    shared_bytes = (shared_bytes // 256) * 256
    actual_blocks_per_sm = shared_memory_per_sm // shared_bytes
    occupancy = actual_blocks_per_sm / full_occupancy_bsm * 100
    print('occupancy', occupancy)
    print('shared_bytes', shared_bytes)

    template = jinja2.Template(code_template, undefined=jinja2.StrictUndefined)
    block = 32
    grid = 1024
    ilp = 1
    its = 400000000 // grid * 32 // block
    its = (its // 256 // ilp) * 256 * ilp
    its = its * blocks_per_sm // full_occupancy_bsm
    name = 'kernel_bsm{bsm}'.format(bsm=blocks_per_sm)
    clearComputeCache()
    add_args = []
    if shared_bytes > 0:
        add_args.append(cl.LocalMemory(shared_bytes))
    source = template.render(name=name, its=its, type='float', **experiment['template_args'], shared=shared_bytes > 0)
    try:
        kernel = buildKernel(name, source)
        for it in range(3):
            t = timeKernel(name, kernel, grid_x=grid, block_x=block, add_args=add_args)
        # print(getPtx(name))
    except Exception as e:
        print(e)
        break

    flops = its * block / (t/1000) * 2 * grid
    times.append({'name': name, 'time': t, 'flops': flops})

print('name\t\t\ttot ms\tgflops')
for time_info in times:
    print('%s\t%.1f\t%.0f' % (time_info['name'].ljust(23), time_info['time'], time_info.get('flops', '') / 1000 / 1000 / 1000))


