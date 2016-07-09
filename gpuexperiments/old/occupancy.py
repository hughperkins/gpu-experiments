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

def buildKernel(name, source):
    options = '-cl-opt-disable'
    # options = ''
    return cl.Program(ctx, source).build(options=options).__getattr__(name) 

d = np.zeros((1024*1024 * 32 * 2,), dtype=np.float32)
d_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=d)

def timeKernel(name, grid_x, kernel):
    # clearComputeCache()
    grid = (grid_x,1,1)
    block = (32,1,1)
    q.finish()
    inittime()
    call_cl_kernel(kernel, q, grid, block, d_cl)
    q.finish()
    return timecheck(name)
    # print(getPtx('mykernel'))

times = []

code_template = r"""
kernel void {{name}} (global int *data) {
    local float F[{{size}}];
    F[0] = 123;
    for(int i = 0; i < 10000; i++) {
    }
}
"""

shared = 0 # 1024
template = jinja2.Template(code_template, undefined=jinja2.StrictUndefined)
while shared < 256:
#    source = code_template
    name = 'shared_%s' % shared
    clearComputeCache()
    size = shared * 1024 // 4
    if size == 0:
        size = 1
    source = template.render(name=name, size=size)
    try:
        kernel = buildKernel(name, source)
        for it in range(3):
            t = timeKernel(name, 1024, kernel)
        print(getPtx(name))
    except Exception as e:
        print(e)
        break
    # print('source', source)
    times.append({'name': name, 'time': t})
    if shared == 0:
        shared = 1
    else:
        shared *= 2

private_template = r"""
kernel void {{name}} (global int *data) {
    local float foo[{{size}}];
    {% for i in range(0, size) %}
    private int i{{i}};
    {% endfor %}
    {% for i in range(0, size) %}
    foo[0] = i{{i}};
    {% endfor %}
    int sum = 0;
    for(int i = 0; i < 10000; i++) {
        sum += data[i];
    }
    data[0] = sum;
}
"""

private_template1 = r"""
kernel void {{name}} (global int *data) {
    {% for i in range(0, size) %}
    private int i{{i}};
    {% endfor %}
    {% for i in range(0, size) %}
    i{{i}} = data[{{i}}];
    {% endfor %}
    int sum = 0;
    {% for i in range(0, size) %}
        sum += i{{i}};
    {% endfor %}
    data[0] = sum;
    {% if addloop %}
    sum = 0;
    for(int i = 0; i < 10000; i++) {
        sum += data[i];
    }
    data[1] = sum;
    {% endif %}
}
"""

private_template2 = r"""
void myfunc(
    {% for i in range(0, size) %}
      int i{{i}},
    {% endfor %}
    int ipad
) {
}

kernel void {{name}} (global int *data) {
    {% for i in range(0, size) %}
    private int i{{i}} = data[{{i}}];
    {% endfor %}
    myfunc(
        {% for i in range(0, size) %}
        i{{i}}, 
        {% endfor %}
        0
    );
    int sum = 0;
    {% for i in range(0, size) %}
        sum += i{{i}};
    {% endfor %}
    data[0] = sum;
    {% if addloop %}
    sum = 0;
    for(int i = 0; i < 10000; i++) {
        sum += data[i];
    }
    data[1] = sum;
    {% endif %}
}
"""

template = jinja2.Template(private_template2, undefined=jinja2.StrictUndefined)
for add_loop in [False, True]:
    private = 0
    while private <= 2048:
    #    source = code_template
        add_loop_str = '_looped'
        if not add_loop:
            add_loop_str = ''
        name = 'private%s_%s' % (add_loop_str, private)
        clearComputeCache()
        size = private
        if size == 0:
            size = 1
        source = template.render(name=name, size=size, addloop=add_loop)
        # print('source', source)
        try:
            kernel = buildKernel(name, source)
            for it in range(3):
                t = timeKernel(name, 1024, kernel)
        except Exception as e:
            print(e)
            break
        # print('source', source)
        print(getPtx(name))
        print(dumpSass(name))
        times.append({'name': name, 'time': t})
        if private == 0:
            private = 1
        else:
            private *= 2

for time_info in times:
    print(time_info['name'], time_info['time'])

