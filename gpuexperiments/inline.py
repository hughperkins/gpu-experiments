# Note that this will erase your nvidia cache, ~/.nv/ComputeCache  This may or may not be an undesirable side-effect for you.  For example, cutorch will take 1-2 minutes or so to start after this cache has been emptied.

from __future__ import print_function, division
import time
import string
import random
import argparse
import numpy as np
import jinja2
import pyopencl as cl
import subprocess
import os
from os.path import join
from gpuexperiments.callkernel import call_cl_kernel
from gpuexperiments.timecheck import inittime, timecheck
from lib_clgpuexp import clearComputeCache, getPtx, timeKernel, buildKernel, initClGpu
import lib_clgpuexp
from gpuexperiments.deviceinfo import DeviceInfo


parser = argparse.ArgumentParser()
parser.add_argument('--printptx', type=bool, default=False, help='note that it erases your nv cache')
# parser.add_argument('--exp')
args = parser.parse_args()

initClGpu()
di = DeviceInfo(lib_clgpuexp.device)

template_func = r"""
{{inline}} void add_one(private float *a, private float *b) {
    *a = *a + *b;
}

kernel void {{name}}(global float *data, global float *out) {
    float a = data[0];
    float b = data[1];
    #pragma unroll 256
    for(int i = 0; i < {{its}}; i++) {
        add_one(&a, &b);
    }
    out[0] = a;
}
"""

template_define = r"""
#define add_one(a, b) \
   a = a + b;

kernel void {{name}}(global float *data, global float *out) {
    float a = data[0];
    float b = data[1];
    #pragma unroll 256
    for(int i = 0; i < {{its}}; i++) {
        add_one(a, b);
    }
    out[0] = a;
}
"""

experiments = [
    {'name': 'k_void', 'inline': '', 'source': template_func},
    {'name': 'k_define', 'inline': '', 'source': template_define},
    {'name': 'k_staticinline', 'inline': 'static inline', 'source': template_func}
]

times = []
its = (4000000//256) * 256
for experiment in experiments:
    name = experiment['name']
    template = jinja2.Template(experiment['source'], undefined=jinja2.StrictUndefined)
    if args.printptx:
        clearComputeCache()
    source = template.render(its=its, **experiment)
    # print('source', source)
    kernel = buildKernel(name, source)
    print('built kernel')
    for it in range(2):
        t = timeKernel(name, kernel, grid_x=1, block_x=32)
    t_sum = 0
    avg_over = 20
    for it in range(avg_over):
        t_sum += timeKernel(name, kernel, grid_x=1, block_x=32)
    t = t_sum / avg_over
    flops = its * 32 * 1 / (t/1000) / 1000 / 1000 / 1000
    times.append({'name': name, 'time': t, 'flops': flops})
    # print(getPtx(name))

f = open('/tmp/inline_%s.tsv' % di.deviceSimpleName, 'w')
line = 'name\ttot ms\tgflops'
f.write(line + '\n')
print(line)
for timeinfo in times:
    line = '%s\t%.1f\t%.1f' % (timeinfo['name'], timeinfo['time'], timeinfo['flops'])
    print(line)
    f.write(line + '\n')
f.close()

