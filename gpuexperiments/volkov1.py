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

code_template = r"""
            kernel void {{kernelname}} (global float *data, global float *out) {
                {% for j in range(ilp) %}
                  float a{{j}} = data[2 + {{j}}];
                {% endfor %}
                float b = data[0];
                float c = data[1];
                #pragma unroll {{unroll}}
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

deviceName = lib_clgpuexp.device.get_info(cl.device_info.NAME)
deviceNameSimple = deviceName.replace('GeForce', '').strip().replace(' ', '').lower()

experiments = [
    #{'name': 'k1_nofma_{block}', 'code': code_template, 'options': '', 'template_args': {'fma': False, 'ilp': 1}},
    {'name': 'k1_fma_ilp1_{block}', 'code': code_template, 'fma': True, 'ilp': 1, 'unroll': 64},
    {'name': 'k1_fma_ilp2_{block}', 'code': code_template, 'fma': True, 'ilp': 2, 'unroll': 64},
    {'name': 'k1_fma_ilp3_{block}', 'code': code_template, 'fma': True, 'ilp': 3, 'unroll': 64},
    {'name': 'k1_fma_ilp4_{block}', 'code': code_template, 'fma': True, 'ilp': 4, 'unroll': 64},
    {'name': 'k1_fma_ilp6_{block}', 'code': code_template, 'fma': True, 'ilp': 6, 'unroll': 64},
    {'name': 'k1_fma_ilp8_{block}', 'code': code_template, 'fma': True, 'ilp': 8, 'unroll': 64}
    #{'name': 'k1_nofma_fastmath_{block}', 'code': code_template, 'options': '-cl-fast-relaxed-math', 'template_args': {'fma': False}},
    #{'name': 'k1_fma_fastmath_{block}', 'code': code_template, 'options': '-cl-fast-relaxed-math', 'template_args': {'fma': True}}
]

for experiment in experiments:
    template = jinja2.Template(experiment['code'], undefined=jinja2.StrictUndefined)
    for block in range(128,1024+128,128):
    #    source = code_template
        its = (8000000//256//experiment['ilp']) * 256 * experiment['ilp']
        if block <= 384:
            its *= 2
        if block <= 128:
            its *= 2
        name = experiment['name'].format(block=block)
        clearComputeCache()
        source = template.render(kernelname=name, its=its, **experiment)
        try:
            kernel = buildKernel(name, source)
            for it in range(2):
                t = timeKernel(name, kernel, block_x=block)
            t_sum = 0
            for it in range(3):
                t_sum += timeKernel(name, kernel, block_x=block)
            t = t_sum / 3
            print(getPtx(name))
        except Exception as e:
            print(e)
            break

        flops = its * block / (t/1000) * 2
        times.append({'name': name, 'time': t, 'flops': flops})

with open('/tmp/volkov1_%s.tsv' % deviceNameSimple, 'w') as f:
    line='name\ttot ms\tgflops'
    print(line)
    f.write(line + '\n')
    for time_info in times:
        line = '%s\t%.1f\t%.0f' % (time_info['name'], time_info['time'], time_info.get('flops', '') / 1000 / 1000 / 1000)
        print(line)
        f.write(line + '\n')

