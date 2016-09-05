"""
Dabble in the algorithm in seciton 2 'GEMM for GTX280'
"""
# Note that this will erase your nvidia cache, ~/.nv/ComputeCache  This may or may not be an undesirable side-effect for you.  For example, cutorch will take 1-2 minutes or so to start after this cache has been emptied.

from __future__ import print_function, division
import argparse
import time
import string
import random
import jinja2
import numpy as np
import pyopencl as cl
import subprocess
import os
from os import path
from os.path import join
from gpuexperiments.callkernel import call_cl_kernel
# import gpuexperiments.cpu_check
from gpuexperiments.timecheck import inittime, timecheck
from gpuexperiments import lib_clgpuexp
from gpuexperiments.lib_clgpuexp import clearComputeCache, getPtx, timeKernel, buildKernel, initClGpu


script_dir = path.dirname(__file__)
basename = path.basename(__file__).split('.')[0]

parser = argparse.ArgumentParser()
parser.add_argument('--printptx', type=bool, default=False)
args = parser.parse_args()

initClGpu()

times = []

GlobalRows = 1024
GlobalMids = 1024
GlobalCols = 1024

blockRows = 32
blockMids = 8
blockCols = 32
# NTBY = 1

BlockRows = GlobalRows // blockRows
BlockMids = GlobalMids // blockMids
BlockCols = GlobalCols // blockCols

print('BlockXs', BlockRows, BlockMids, BlockCols)

# global dimensions:
#   GlobalRows GlobalMids GlobalCols
# block dimensions:
#   blockRows blockCols blockMids
# count of blocks
#   BlockRows BlockMids BlockCols
# global block id:
#   BlockRow BlockMid BlockCol
# local block idx:
#   blockRow blockMid blockCol
# data:
#   A B C
#   Ablk Bblk Cblk

with open(join(script_dir, 'gtx280_v4.jinja2.cl')) as f:
    code_template = f.read()

template = jinja2.Template(code_template, undefined=jinja2.StrictUndefined)
kernelname = 'gtx280'
source = template.render(
    kernelname=kernelname, blockCols=blockCols, blockMids=blockMids,
    blockRows=blockRows)

np.random.seed(123)
A = np.random.randn(GlobalRows, GlobalMids).astype(np.float32)
B = np.random.randn(GlobalMids, GlobalCols).astype(np.float32)
C = np.zeros((GlobalRows, GlobalCols), dtype=np.float32)

mf = lib_clgpuexp.mf
ctx = lib_clgpuexp.ctx
q = lib_clgpuexp.q
# cl = lib_clgpuexp.cl

A_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=A)
B_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=B)
C_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=C)

C_cpu = A.dot(B)
print('C_cpu', C_cpu)

kernel = cl.Program(ctx, source).build(options='').__getattr__(kernelname)
grid = (BlockRows, BlockCols)
block = (blockRows, 1)
print('grid', grid, 'block', block)
for it in range(3):
    call_cl_kernel(
        kernel, q, grid, block,
        GlobalRows, GlobalMids, GlobalCols,
        BlockRows, BlockMids, BlockCols,
        blockRows, blockMids, blockCols,
        C_cl, A_cl, B_cl,
        cl.LocalMemory(blockMids * blockCols * 4), cl.LocalMemory(blockRows * blockMids * 4)
    )
q.finish()
start = time.time()
its = 20
for it in range(its):
    call_cl_kernel(
        kernel, q, (BlockRows, BlockCols), (blockRows, 1),
        GlobalRows, GlobalMids, GlobalCols,
        BlockRows, BlockMids, BlockCols,
        blockRows, blockMids, blockCols,
        C_cl, A_cl, B_cl,
        cl.LocalMemory(blockMids * blockCols * 4), cl.LocalMemory(blockRows * blockMids * 4 * 0)
    )
q.finish()
end = time.time()
diff = end - start
avg_time = diff / its
flops = GlobalRows * GlobalRows * GlobalCols * 2
# print('flops', flops)
C_gpu = C.copy()
cl.enqueue_copy(q, C_gpu, C_cl)
q.finish()

print('C_gpu', C_gpu)
delta = np.abs(C_gpu - C_cpu).max()

print('')
print('total time     %.4fs; per iteration %.3fs' % (diff, avg_time))
gigabytes = (GlobalRows * GlobalMids * 4 + GlobalMids * GlobalCols * 4 + GlobalRows * GlobalCols * 4) / 1000 / 1000 / 1000
print('to/from global %.3f GB/s' % (gigabytes / avg_time))
gigabytes_cores = (GlobalRows * GlobalCols * GlobalMids * 4 * 3 / 1000 / 1000 / 1000)  # multiply by 3 because for each
# iteration through inner loop, need to get value of A to core, value of B to core, and retrieve value of C
print('to/from cores  %.1f GB/s' % (gigabytes_cores / avg_time))
print('flops          %.1f GFLOPS/s' % (flops / avg_time / 1000 / 1000 / 1000))

assert delta <= 1e-3
