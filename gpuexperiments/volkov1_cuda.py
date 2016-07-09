# Note that this will erase your nvidia cache, ~/.nv/ComputeCache  This may or may not be an undesirable side-effect for you.  For example, cutorch will take 1-2 minutes or so to start after this cache has been emptied.

from __future__ import print_function, division
import time
import string
import random
import jinja2
import json
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import subprocess
import os
from os import path
from os.path import join
from gpuexperiments.callkernel_cuda import call_cuda_kernel
#import gpuexperiments.cpu_check
from gpuexperiments.timecheck import inittime, timecheck

gpu_idx = 0


if not path.isdir('/tmp/cudaptx'):
    os.makedirs('/tmp/cudaptx')

def buildKernel(name, source, options=''):
    # options = '-cl-opt-disable'
    # options = ''
    return SourceModule(source, keep=True, cache_dir='/tmp/cudaptx').get_function(name) 

d = np.zeros((1024*1024 * 32 * 2,), dtype=np.float32)
d_cuda = gpuarray.to_gpu(d)

out = np.zeros((1024,), dtype=np.float32)
out_cuda = gpuarray.to_gpu(out)

def timeKernel(name, kernel, grid_x=1, block_x=32):
    # clearComputeCache()
    grid = (grid_x,1,1)
    block = (block_x,1,1)
    cuda.Context.synchronize()
    inittime()
    call_cuda_kernel(kernel, grid, block, d_cuda, out_cuda)
    cuda.Context.synchronize()
    return timecheck(name)
    # print(getPtx('mykernel'))

times = []

code_template = r"""
            __global__ void {{name}} (float *data, float *out) {
                float a = data[0];
                float b = data[1];
                float c = data[2];
                #pragma unroll 256
                for(int i = 0; i < {{its}}; i++) {
                    {% if fma %}
                    a = fma(a, b, c);
                    {% else %}
                    a = a * b + c;
                    {% endif %}
                }
                out[0] = a;
            }
        """

code_template_nopragma = r"""
            __global__ void {{name}} (float *data, float *out) {
                float a = data[0];
                float b = data[1];
                float c = data[2];
                for(int i = 0; i < {{its}}; i+= {{unroll}}) {
                    {% for j in range(unroll) %}
                    {% if fma %}
                    a = fma(a, b, c);
                    {% else %}
                    a = a * b + c;
                    {% endif %}
                    {% endfor %}
                }
                out[0] = a;
            }
        """
256
experiments = [
    {'name': 'k1_nofma_{block}', 'kernelname': 'k1_nofma', 'code': code_template, 'options': '', 'template_args': {'fma': False}},
    {'name': 'k1_fma_{block}', 'kernelname': 'k1_fma', 'code': code_template, 'options': '', 'template_args': {'fma': True}}
    # {'name': 'k1_nofma_fastmath_{block}', 'kernelname': 'k1_nofma_fastmath', 'code': code_template, 'options': '-cl-fast-relaxed-math', 'template_args': {'fma': False}},
    # {'name': 'k1_fma_fastmath_{block}', 'kernelname': 'k1_fma_fastmath', 'code': code_template, 'options': '-cl-fast-relaxed-math', 'template_args': {'fma': True}}
]

its = (4000000//256) * 256
kernel_by_source = {}
for experiment in experiments:
    template = jinja2.Template(experiment['code'], undefined=jinja2.StrictUndefined)
    source = template.render(name=experiment['kernelname'], its=its, **experiment['template_args'])
    build_info = {'source': source, 'options': experiment['options']}
    build_info_str = json.dumps(build_info)
    if build_info_str not in kernel_by_source:
        kernel = buildKernel(source=source, name=experiment['kernelname'], options=experiment['options'])
        kernel_by_source[build_info_str] = kernel
    kernel = kernel_by_source[build_info_str]
    for block in range(128,1024+128,128):
    #    source = code_template
        name = experiment['name'].format(block=block)
        # clearComputeCache()
        try:
            for it in range(3):
                t = timeKernel(name, kernel, block_x=block)
            # print(getPtx(name))
        except Exception as e:
            print(e)
            break

        flops = its * block / (t/1000) * 2
        times.append({'name': name, 'time': t, 'flops': flops})


print('kernel time(ms) GFLOPS')
for time_info in times:
    print('%s %s %s' % (time_info['name'], time_info['time'], time_info.get('flops', '') / 1000 / 1000 / 1000))

