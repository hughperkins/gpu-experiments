# Note that this will erase your nvidia cache, ~/.nv/ComputeCache  This may or may not be an undesirable side-effect for you.  For example, cutorch will take 1-2 minutes or so to start after this cache has been emptied.

from __future__ import print_function, division
import time
import string
import random
import traceback
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
from lib_clgpuexp import dumpSass


initClGpu()

times = []

code_template = r"""
            kernel void {{name}} (global float *data, global float *out) {
                float a = data[0];
                float b = data[1];
                float c = data[2];
                #pragma unroll 4
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

code_template_nopragma2 = r"""
            kernel void {{name}} (global float *data, global float *out) {
                float a = data[0];
                float b = data[1];
                float c = data[2];
                for(int i = - ({{its}} / {{unroll}}); i != 0; i++) {
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
    #{'name': 'k1_noopt_{block}', 'code': code_template, 'options': '-cl-opt-disable', 'template_args': {'fma': False}},
    {'name': 'k1_opt_{block}', 'code': code_template, 'options': '', 'template_args': {'fma': False}},
    {'name': 'k1_noprag4_noopt_{block}', 'code': code_template_nopragma, 'options': '-cl-opt-disable', 'template_args': {'fma': False, 'unroll': 4}},
    {'name': 'k1_noprag4b_noopt_{block}', 'code': code_template_nopragma2, 'options': '-cl-opt-disable', 'template_args': {'fma': False, 'unroll': 4}}
    #{'name': 'k1_noprag128_noopt_{block}', 'code': code_template_nopragma, 'options': '-cl-opt-disable', 'template_args': {'fma': False, 'unroll': 128}}
    #{'name': 'k1_noprag128_opt_{block}', 'code': code_template_nopragma, 'options': '', 'template_args': {'fma': False, 'unroll': 128}}
    #{'name': 'k1_fma_noopt_{block}', 'code': code_template, 'options': '-cl-opt-disable', 'template_args': {'fma': True}},
    #{'name': 'k1_fma_opt_{block}', 'code': code_template, 'options': '', 'template_args': {'fma': True}}
]

its = (1000000//256) * 256
for experiment in experiments:
    template = jinja2.Template(experiment['code'], undefined=jinja2.StrictUndefined)
    for block in [128]: # range(128,1024+128,128):
    #    source = code_template
        name = experiment['name'].format(block=block)
        clearComputeCache()
        source = template.render(name=name, its=its, **experiment['template_args'])
        try:
            kernel = buildKernel(name, source, options=experiment['options'])
            for it in range(3):
                t = timeKernel(name, kernel, block_x=block)
            print(getPtx(name))
            dumpSass(name)
        except Exception as e:
            print(e)
            break

        flops = its * block / (t / 1000) * 2
        times.append({'name': name, 'time': t, 'flops': flops})


print('kernel\t\t\ttot ms\tgflops')
for time_info in times:
    print('%s\t%.1f\t%.2f' % (time_info['name'].ljust(23), time_info['time'], time_info.get('flops', '') / 1000 / 1000 / 1000))

