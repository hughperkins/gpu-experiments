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


times = []

code_template = r"""
            kernel void {{name}} (global float *data, global float *out) {
                int tid = get_local_id(0);
                local {{type}} F[{{shared}}];
                F[tid] = data[tid];
                {% for j in range(ilp) %}
                  float a{{j}} = F[{{j}}];
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
    {'name': 'k1_g{grid}_b{block}_s{shared}', 'code': code_template, 'options': '', 'template_args': {'fma': True, 'ilp': 1}, 'block': 32, 'grid': 1},
    {'name': 'k1_g{grid}_b{block}_s{shared}', 'code': code_template, 'options': '', 'template_args': {'fma': True, 'ilp': 1}, 'block': 1024, 'grid': 1},
    {'name': 'k1_g{grid}_b{block}_s{shared}', 'code': code_template, 'options': '', 'template_args': {'fma': True, 'ilp': 1}, 'block': 32, 'grid': 1024},
    {'name': 'k1_g{grid}_b{block}_s{shared}', 'code': code_template, 'options': '', 'template_args': {'fma': True, 'ilp': 1}, 'block': 64, 'grid': 1024},
    {'name': 'k1_g{grid}_b{block}_s{shared}', 'code': code_template, 'options': '', 'template_args': {'fma': True, 'ilp': 1}, 'block': 128, 'grid': 1024},
    {'name': 'k1_g{grid}_b{block}_s{shared}', 'code': code_template, 'options': '', 'template_args': {'fma': True, 'ilp': 1}, 'block': 32, 'grid': 4096},
    {'name': 'k1_g{grid}_b{block}_s{shared}', 'code': code_template, 'options': '', 'template_args': {'fma': True, 'ilp': 1}, 'block': 64, 'grid': 4096},
    {'name': 'k1_g{grid}_b{block}_s{shared}', 'code': code_template, 'options': '', 'template_args': {'fma': True, 'ilp': 1}, 'block': 32, 'grid': 16},
    {'name': 'k1_g{grid}_b{block}_s{shared}', 'code': code_template, 'options': '', 'template_args': {'fma': True, 'ilp': 1}, 'block': 32, 'grid': 32},
    {'name': 'k1_g{grid}_b{block}_s{shared}', 'code': code_template, 'options': '', 'template_args': {'fma': True, 'ilp': 1}, 'block': 32, 'grid': 48}
]

initClGpu()

compute_units = lib_clgpuexp.device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
maxShared = lib_clgpuexp.device.get_info(cl.device_info.LOCAL_MEM_SIZE) // 1024
print('compute units', compute_units, 'max shared memory', maxShared)
#sys.exit(1)
for experiment in experiments:

    template = jinja2.Template(experiment['code'], undefined=jinja2.StrictUndefined)
    #for block in range(128,1024+128,128):
    block = experiment['block']
    grid = experiment['grid'] # 1024
    ilp = experiment['template_args']['ilp']
    its = 40000000 // grid * 32 // block
    if grid == 1:  # modify its, since it will only run on one sm
        its = its // compute_units
    its = (its // 256 // ilp) * 256 * ilp
    for shared in range(0, maxShared, 4):
        if shared == 0 and block > 32:
            continue # since we cant actually copy the global meory down into a zero-sized shared memory
    #    source = code_template
        name = experiment['name'].format(shared=shared, grid=grid, block=block)
        clearComputeCache()
        shared_floats = shared * 1024 // 4
        if shared_floats == 0:
            shared_floats = 32
        source = template.render(name=name, its=its, type='float', **experiment['template_args'], shared=shared_floats)
        # print('source', source)
        try:
            kernel = buildKernel(name, source, options=experiment['options'])
            for it in range(3):
                t = timeKernel(name, kernel, grid_x=grid, block_x=block)
            print(getPtx(name))
        except Exception as e:
            print(e)
            break

        flops = its * block / (t/1000) * 2 * grid
        times.append({'name': name, 'time': t, 'flops': flops})


print('name\t\t\ttot ms\tgflops')
for time_info in times:
    print('%s\t%.1f\t%.0f' % (time_info['name'].ljust(23), time_info['time'], time_info.get('flops', '') / 1000 / 1000 / 1000))

