# Note that this will erase your nvidia cache, ~/.nv/ComputeCache  This may or may not be an undesirable side-effect for you.  For example, cutorch will take 1-2 minutes or so to start after this cache has been emptied.

from __future__ import print_function, division
import time
import argparse
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


parser = argparse.ArgumentParser()
parser.add_argument('--printptx', type=bool, default=False)
parser.add_argument('--exp')
args = parser.parse_args()

initClGpu()
deviceName = lib_clgpuexp.device.get_info(cl.device_info.NAME)
deviceSimpleName = deviceName.replace('GeForce', '').replace('GTX', '').strip().replace(' ', '').lower()

template = r"""
    kernel void {{kernelname}}(global {{type}} *data, global {{type}} *out) {
        {{type}} a = data[0];
        {{type}} b = data[1];
        {{type}} c = data[2];
        #pragma unroll {{unroll}}
        for(int i = 0; i < {{its}}; i++) {
            a = {{op}};
        }
        out[0] = a;
    }
"""

template_rand = r"""
    kernel void {{kernelname}}(global {{type}} *data, global {{type}} *out) {
        {{type}} a = data[0];
        {{type}} b = data[1];
        {{type}} c = data[2];
        {{type}} d = 0;
        #pragma unroll {{unroll}}
        for(int i = 0; i < {{its}}; i++) {
            //a = {{op}};
            a = (a * c) >> 2;
            // d += b;
            {{op}};
        }
        out[0] = a;
        out[1] = d;
    }
"""

experiments = [
    {'name': '{type}_add', 'op': 'a + b', 'type': 'float', 'code': template, 'ops': 1},
    {'name': '{type}_mul', 'op': 'a * b', 'type': 'float', 'code': template, 'ops': 1},
    {'name': '{type}_sub', 'op': 'a - b', 'type': 'float', 'code': template, 'ops': 1},
    {'name': '{type}_div', 'op': 'a / b', 'type': 'float', 'code': template, 'ops': 1},
    {'name': '{type}_fma', 'op': 'fma(a, b, c)', 'type': 'float', 'code': template, 'ops': 2},
    {'name': '{type}_sqrt', 'op': 'sqrt(a)', 'type': 'float', 'code': template, 'ops': 1},
    {'name': '{type}_native_sqrt', 'op': 'native_sqrt(a)', 'type': 'float', 'code': template, 'ops': 1},
    {'name': '{type}_tanh', 'op': 'tanh(a)', 'type': 'float', 'code': template, 'ops': 1},

    {'name': '{type}_mul', 'op': 'a * b', 'type': 'int', 'code': template, 'ops': 1},
    {'name': '{type}_div', 'op': 'a / b', 'type': 'int', 'code': template, 'ops': 1},
    {'name': '{type}_randbase', 'op': '', 'type': 'int', 'code': template_rand, 'ops': 0},
    {'name': '{type}_randbase_add', 'op': 'a = a + b', 'type': 'int', 'code': template_rand, 'ops': 1},
    {'name': '{type}_randbase_sub', 'op': 'a = a - b', 'type': 'int', 'code': template_rand, 'ops': 1},
    {'name': '{type}_randbase_mul', 'op': 'a = a * b', 'type': 'int', 'code': template_rand, 'ops': 1},
    {'name': '{type}_randbase_div', 'op': 'a = a / b', 'type': 'int', 'code': template_rand, 'ops': 1},
    {'name': '{type}_randbase_shift', 'op': 'a = a << b', 'type': 'int', 'code': template_rand, 'ops': 1},
    #{'name': '{type}_randbase_shiftshiftadd', 'op': 'a = (a << b) + (a << c)', 'type': 'int', 'code': template, 'ops': 1}
    #{'name': '{type}_shiftshiftadd', 'op': '(a << 2) + (a <<1 )', 'type': 'int', 'code': template, 'ops': 1}
    #{'name': '{type}_add', 'op': 'a + b; a = a + c', 'type': 'int', 'code': template, 'ops': 2},
#    {'name': '{type}_sub', 'op': 'a - b', 'type': 'int', 'code': template, 'ops': 1}
]

times = []
unroll = 128
block = 32
for experiment in experiments:
    if args.exp is not None and args.exp not in experiment['name']:
        continue
    name = experiment['name'].format(**experiment)
    print('name', name)
    its = 10000000
    divider = 1
    if name in ['int_div', 'float_tanh', 'int_randbase_div']:
        divider = 30
        its //= divider
    its = (its // unroll) * unroll
    template = jinja2.Template(experiment['code'], undefined=jinja2.StrictUndefined)
    if args.printptx:
        clearComputeCache()
    #template_dict = {}
    #for k, v in experiment.items():
    #    template_dict[k] = v
    #template_dict['name'] = name
    # print(template_dict)
    source = template.render(kernelname=name, its=its, unroll=unroll, **experiment)

    kernel = buildKernel(name, source)
    print('built kernel')
    for it in range(3):
        t = timeKernel(name, kernel, block_x=block)
    if args.printptx:
        print(getPtx(name))
    if 'randbase' in name:
        if name == 'int_randbase':
            randbase_time = t
            continue
        #else:
        t = t - randbase_time / divider
    gflops = 1/ (t / 1000) * experiment['ops'] * block * its / 1000 / 1000 / 1000
    op_ns = t / its * 1000 * 1000
    times.append({'name': name, 'time': t, 'gflops': gflops, 'op_ns': op_ns})

f = open('/tmp/maths2_%s.tsv' % deviceSimpleName, 'w')
line = 'name\ttot ms\top ns\tgflops'
print(line)
f.write(line + '\n')
for timeinfo in times:
    line = '%s\t%.1f\t%.2f\t%.2f' % (timeinfo['name'].ljust(10), timeinfo['time'], timeinfo['op_ns'], timeinfo['gflops'])
    print(line)
    f.write(line + '\n')
f.close()

