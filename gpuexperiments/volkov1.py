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
from lib_clgpuexp import clearComputeCache, getPtx, timeKernel, buildKernel, initClGpu


initClGpu()

times = []

code_template = r"""
            kernel void {{name}} (global float *data, global float *out) {
                {% for j in range(ilp) %}
                  float a{{j}} = data[2 + {{j}}];
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

code_template_nopragma = r"""
            kernel void {{name}} (global float *data, global float *out) {
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

experiments = [
    #{'name': 'k1_nofma_{block}', 'code': code_template, 'options': '', 'template_args': {'fma': False, 'ilp': 1}},
    #{'name': 'k1_fma_{block}', 'code': code_template, 'options': '', 'template_args': {'fma': True, 'ilp': 1}},
    #{'name': 'k1_fma_ilp2_{block}', 'code': code_template, 'options': '', 'template_args': {'fma': True, 'ilp': 2}},
    {'name': 'k1_fma_ilp3_{block}', 'code': code_template, 'options': '', 'template_args': {'fma': True, 'ilp': 3}}
    #{'name': 'k1_nofma_fastmath_{block}', 'code': code_template, 'options': '-cl-fast-relaxed-math', 'template_args': {'fma': False}},
    #{'name': 'k1_fma_fastmath_{block}', 'code': code_template, 'options': '-cl-fast-relaxed-math', 'template_args': {'fma': True}}
]

for experiment in experiments:
    its = (4000000//256//experiment['template_args']['ilp']) * 256 * experiment['template_args']['ilp']
    template = jinja2.Template(experiment['code'], undefined=jinja2.StrictUndefined)
    for block in range(128,1024+128,128):
    #    source = code_template
        name = experiment['name'].format(block=block)
        clearComputeCache()
        source = template.render(name=name, its=its, **experiment['template_args'])
        try:
            kernel = buildKernel(name, source, options=experiment['options'])
            for it in range(3):
                t = timeKernel(name, kernel, block_x=block)
            print(getPtx(name))
        except Exception as e:
            print(e)
            break

        flops = its * block / (t/1000) * 2
        times.append({'name': name, 'time': t, 'flops': flops})


print('name\t\t\ttot ms\tgflops')
for time_info in times:
    print('%s\t%.1f\t%.0f' % (time_info['name'].ljust(23), time_info['time'], time_info.get('flops', '') / 1000 / 1000 / 1000))

