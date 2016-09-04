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
from os.path import join
from gpuexperiments.callkernel import call_cl_kernel
# import gpuexperiments.cpu_check
from gpuexperiments.timecheck import inittime, timecheck
from gpuexperiments import lib_clgpuexp
from gpuexperiments.lib_clgpuexp import clearComputeCache, getPtx, timeKernel, buildKernel, initClGpu


parser = argparse.ArgumentParser()
parser.add_argument('--printptx', type=bool, default=False)
args = parser.parse_args()

initClGpu()

times = []

GlobalRows = 1024
GlobalMids = 1024
GlobalCols = 1024

blockRows = 32
blockMids = 16
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

code_template = r"""
    kernel void {{kernelname}} (
            int GlobalRows, int GlobalMids, int GlobalCols,
            int BlockRows, int BlockMids, int BlockCols,
            int blockRows, int blockMids, int blockCols,
            global float *C, global float *A, global float *B,
            local float *B_block) {
        int BlockRow = get_group_id(0);
        int BlockCol = get_group_id(1);
        int tid = get_local_id(0);
        // int blockRow = get_local_id(0);
        int globalRow = BlockRow * blockRows + tid;

        float C_row[{{blockCols}}];
        //float C_row1[{{blockCols}} >> 1];
        for(int blockCol=0; blockCol < blockCols; blockCol++) {
             C_row[blockCol] = 0.0f;
          //  C_row1[blockCol] = 0.0f;
        }
        {
            int blockCol = tid;
            int globalCol = BlockCol * blockCols + blockCol;
            for(int BlockMid = 0; BlockMid < BlockMids; BlockMid++) {
                // first copy down the data from B
                // each thread will handle one column of B data, ie
                // iterate over blockMid
                // sync point (can remove if num threads == warpsize)
                //barrier(CLK_LOCAL_MEM_FENCE);
                for(int blockMid=0; blockMid < blockMids; blockMid++) {
                    int globalMid = BlockMid * blockMids + blockMid;
                    B_block[blockMid * blockCols + blockCol] = B[globalMid * GlobalCols + globalCol];
                }
                // should probably copy down A too?  (otherwise have to wait for each float of A to come down,
                // one by one...)
                // but lets copy to private for now, no coasllescing, then try coallescing in v0.2
                float A_row[{{blockMids}}];
                for(int blockMid=0; blockMid < blockMids; blockMid++) {
                    int globalMid = BlockMid * blockMids + blockMid;
                    A_row[blockMid] = A[globalRow * GlobalMids + globalMid];
                }
                // sync point (can remove if num threads == warpsize)
                //barrier(CLK_LOCAL_MEM_FENCE);

                // calc some C :-)
                // each thread handles a row of c, so needs to iterate over columns
                // but for each column, needs to iterate over middle too
                    for(int blockCol=0; blockCol < blockCols; blockCol++) {
                for(int blockMid=0; blockMid < blockMids; blockMid++) {
                    int blockMid_blockCols = blockMid * blockCols;
                    float a = A_row[blockMid];
                        C_row[blockCol] += a * B_block[blockMid_blockCols + blockCol];
                        // C_row1[blockCol] += A_row[blockMid + 1] * B_block[blockMid * blockCols + blockCol];
                    }
                }
            }
        }
        // write C out
        int globalRow_globalCols = globalRow * GlobalCols;
        int BlockCol_blockCols = BlockCol * blockCols;
        int globalRow_globalCols_plus_BlockCol_blockCols = globalRow_globalCols + BlockCol_blockCols;
        for(int blockCol=0; blockCol < blockCols; blockCol++) {
            //int globalCol = BlockCol_blockCols + blockCol;
            C[globalRow_globalCols_plus_BlockCol_blockCols + blockCol] = C_row[blockCol];
        }
    }
"""

template = jinja2.Template(code_template, undefined=jinja2.StrictUndefined)
kernelname = 'gtx280'
source = template.render(kernelname=kernelname, blockCols=blockCols, blockMids=blockMids)

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
        cl.LocalMemory(blockMids * blockCols * 4)
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
        cl.LocalMemory(blockMids * blockCols * 4)
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
print('total time     %.2fs; per iteration %.3fs' % (diff, avg_time))
gigabytes = (GlobalRows * GlobalMids * 4 + GlobalMids * GlobalCols * 4 + GlobalRows * GlobalCols * 4) / 1000 / 1000 / 1000
print('to/from global %.3f GB/s' % (gigabytes / avg_time))
gigabytes_cores = (GlobalRows * GlobalCols * GlobalMids * 4 / 1000 / 1000 / 1000)
print('to/from cores  %.1f GB/s' % (gigabytes_cores / avg_time))
print('flops          %.1f GFLOPS/s' % (flops / avg_time / 1000 / 1000 / 1000))

assert delta <= 1e-3
