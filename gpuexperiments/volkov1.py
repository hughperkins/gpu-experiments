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

gpu_idx = 0

platforms = cl.get_platforms()
i = 0
for platform in platforms:
   gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
   if gpu_idx < i + len(gpu_devices):
       ctx = cl.Context(devices=[gpu_devices[gpu_idx-i]])
       break
   i += len(gpu_devices)

print('context', ctx)
q = cl.CommandQueue(ctx)  

mf = cl.mem_flags

def clearComputeCache():
    cache_dir = join(os.environ['HOME'], '.nv/ComputeCache')
    for subdir in os.listdir(cache_dir):
        if subdir == 'index':
            continue
        print('clean', subdir)
        subprocess.call(['rm', '-Rf', join(cache_dir, subdir)])
#    subprocess.call(['rm', '-Rf', join(os.environ['HOME'], '.nv/ComputeCache')])

def getPtx(kernelName):
    with open('/tmp/gpucmd.sh', 'w') as f:
        f.write(r"""#!/bin/bash
        cat $(grep -r %s ~/.nv/ComputeCache | awk '{print $3}')
"""  % kernelName)
    filepath = subprocess.check_output(['/bin/bash', '/tmp/gpucmd.sh'])
    filepath_utf8 = ''
    for byte in filepath:
        # print(byte)
        if byte >= 10 and byte < 128:
           if chr(byte) in string.printable:
               filepath_utf8 += chr(byte)
    # print('filepath', filepath)
    #print(kernelName)
    ptx = filepath_utf8.split('--opt-level')[0]
    print(ptx)
    return ptx

def dumpSass(kernelName):
    ptx = getPtx(kernelName)
    ptx = ptx.split('.version 5.0')[1].split('A')[0]
    ptx = '.version 4.3\n' + ptx
    # print('ptx', ptx)
    #sys.exit(1)
    with open('/tmp/~kernel.ptx', 'w') as f:
        f.write(ptx)
    print(subprocess.check_output([
        'ptxas',
        '--gpu-name', 'sm_50',
        '--output-file', '/tmp/~kernel.o',
        '/tmp/~kernel.ptx']).decode('utf-8'))
    sass = subprocess.check_output([
        'cuobjdump', '--dump-sass', '/tmp/~kernel.o']).decode('utf-8')
    print(sass)
    return sass

def buildKernel(name, source, options=''):
    # options = '-cl-opt-disable'
    # options = ''
    return cl.Program(ctx, source).build(options=options).__getattr__(name) 

d = np.zeros((1024*1024 * 32 * 2,), dtype=np.float32)
d_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=d)

out = np.zeros((1024,), dtype=np.float32)
out_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=out)

def timeKernel(name, kernel, grid_x=1, block_x=32):
    # clearComputeCache()
    grid = (grid_x,1,1)
    block = (block_x,1,1)
    q.finish()
    inittime()
    call_cl_kernel(kernel, q, grid, block, d_cl, out_cl)
    q.finish()
    return timecheck(name)
    # print(getPtx('mykernel'))

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


print('kernel time(ms) GFLOPS')
for time_info in times:
    print('%s %s %s' % (time_info['name'], time_info['time'], time_info.get('flops', '') / 1000 / 1000 / 1000))

