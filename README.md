# gpu-experiments

## Specific

Informal experiments on various gpu kernel questions

With varying level of rigorousness...

Approximate target kernel I'm pondering as I write this https://github.com/hughperkins/neonCl-underconstruction/blob/c80492bd1fc5fd2e33ef3ad06f601a39a68ce9b3/winograd_kernels_cl.py#L317

Terminology will be interchangeably cuda/opencl.  Experiments will run on different devices.  Where not specified, they are running on NVIDIA 940M, which is a Maxwell.

### Effect of optimization on performance?

[gpuexperiments/optimization2.py](gpuexperiments/optimization2.py)

```
kernel time(ms) GFLOPS
k1_noopt_128 140.652179718018 1.81997617465439
k1_opt_128 9.61518287658691 26.6228546337193
k1_noprag_noopt_128 16.4401531219482 15.5706345373543
k1_noprag_opt_128 6.25729560852051 40.9096248627649
k1_fma_noopt_128 483.375072479248 0.529575542005199
k1_fma_opt_128 9.49525833129883 26.9590996967625
```
- looks like optimizatoin makes a huge difference. Why?
- unroll is part of the story: `#pragma unroll` seems to do nothing, when optimization is turned off
  - the `noprag` kernels use jinja2 to unroll, rather than using `#pragma unroll`
  - but still slower, so not the whole story

Going right down to the SASS, no obious differences.  For optimized, the sass for the loop is:
```
        /*0058*/                   IADD32I R4, R4, 0x4;                       /* 0x1c00000000470404 */
                                                                              /* 0x301fd800fe2d07f5 */
        /*0068*/                   FFMA R2, R7, R0.reuse, R5.reuse;           /* 0x5980028000070702 */
        /*0070*/                   ISETP.NE.AND P0, PT, R4, RZ, PT;           /* 0x5b6b03800ff70407 */
        /*0078*/                   FFMA R2, R2, R0.reuse, R5.reuse;           /* 0x5980028000070202 */
                                                                              /* 0x001ff400fe0c07f6 */
        /*0088*/                   FFMA R2, R2, R0.reuse, R5.reuse;           /* 0x5980028000070202 */
        /*0090*/         {         FFMA R7, R2, R0, R5;                       /* 0x5980028000070207 */
        /*0098*/               @P0 BRA 0x58;        }                         /* 0xe2400ffffb80000f */
```
For un-optimized, it is:
```
        /*0050*/                   IADD32I R4, R4, 0x4;                                /* 0x1c00000000470404 */
        /*0058*/                   FFMA R2, R7, R0.reuse, R5.reuse;                    /* 0x5980028000070702 */
                                                                                       /* 0x301fd980fec007f1 */
        /*0068*/                   ISETP.LT.AND P0, PT, R4, c[0x2][0x0], PT;           /* 0x4b63038800070407 */
        /*0070*/                   FFMA R2, R2, R0.reuse, R5.reuse;                    /* 0x5980028000070202 */
        /*0078*/                   FFMA R2, R2, R0.reuse, R5.reuse;                    /* 0x5980028000070202 */
                                                                                       /* 0x001fc400ffa007f0 */
        /*0088*/         {         FFMA R7, R2, R0, R5;                                /* 0x5980028000070207 */
        /*0090*/               @P0 BRA 0x50;        }                                  /* 0xe2400ffffb80000f */
```
Basically the same?  Just one has `c[0x2][0x0]` and one has `RZ`.  Could be the reason though, so modified the nopragma kernel, code nopragma2, which gave sass:
```
        /*0050*/                   IADD32I R4, R4, 0x1;                       /* 0x1c00000000170404 */
        /*0058*/                   FFMA R2, R7, R0.reuse, R5.reuse;           /* 0x5980028000070702 */
                                                                              /* 0x301fd980fec007f1 */
        /*0068*/                   ISETP.NE.AND P0, PT, R4, RZ, PT;           /* 0x5b6b03800ff70407 */
        /*0070*/                   FFMA R2, R2, R0.reuse, R5.reuse;           /* 0x5980028000070202 */
        /*0078*/                   FFMA R2, R2, R0.reuse, R5.reuse;           /* 0x5980028000070202 */
                                                                              /* 0x001fc400ffa007f0 */
        /*0088*/         {         FFMA R7, R2, R0, R5;                       /* 0x5980028000070207 */
        /*0090*/               @P0 BRA 0x50;        }     
```
Exactly identical to optimized version?  But 5 times slower:
```
kernel			tot ms	gflops
k1_opt_128             	9.4	27.17
k1_noprag4_noopt_128   	47.6	5.38
k1_noprag4b_noopt_128  	46.5	5.50
```

Without any way of explaining this discrepancy, it looks like we should do all experiments with optimizations on unfortunately.  This makes it harder to make artificial kernels for timing that dont just get entirely optimized away.

### inlining?

Do functions get inlined?  When?  [gpuexperiments/inline.py](gpuexperiments/inline.py)

```
name		tot ms	gflops
k_staticinline	24.5	5.2
k_void	24.7	5.2
k_define	24.7	5.2
```
With optimizations on, `#define`, `static inline`, or normal funcion, all run at the exact same speed.

### maths

[gpuexperiments/maths2.py](gpuexperiments/maths2.py)

Comparison of maths operators.  This is using a single block of 32 threads.  Results on 940M:

<img src="img/maths2_940m.png?raw=true" width="600" height="400" />

[results/maths2_940m.tsv](results/maths2_940m.tsv)

`int add` and `int sub` were optimized away, so removed from the graph.  Not sure how to work around that?  Section 'effect of optimization on performance?' shows that just turning off optimizations gives unrepresentative results, so not really an option.

For other operations, we see:
- `fma` is the fastest
- float add/sub/mul all same speed as each other
- float div about 3 times slower than mul
- int mul half as fast as float mul
- int div slllooowwww
- float sqrt is pretty slow, but still faster than int div

### Occupancy

In Volkov, slide 30, he suggests that we can allocated shared memory dynamically, to control occupancy.  Let's try this:

[gpuexperiments/occupancy_dyn.py](gpuexperiments/occupancy_dyn.py)

For 940M:

<img src="img/occupancy_by_shared_940m.png?raw=true" width="600" height="400" />

[results/occupancy_dyn_940m.tsv](results/occupancy_dyn_940m.tsv)

We can see that using dynamically allocated shared memory, on NVIDIA, can indeed control the occupancy, doesnt get optimized away.

For Titan X:

<img src="img/occupancy_by_shared_titanx.png?raw=true" width="600" height="400" />

[results/occupancy_dyn_titanx.tsv](results/occupancy_dyn_titanx.tsv)

Again, a similar graph.  Strangely, with no shared memory, the performance is slightly worse, which is mysterious.

In both cases, using blocksize 64, instead of 32, improves flops somewhat, which we would expect, since blocksize 32 should only get a maximum occupancy of 50%.  Why? Because on sm5.0 and sm5.2, there is a maximum of 32 blocks per SM, which with a blocksize of 32 is a maximum of 32*32 = 1024 threads.  However, the maximum threads on sm5.0 and sm5.2 is 2048, so we are not getting 100% occupancy with blocksize 32.  Blocksize 64 should be able to obtain 100% occupancy.  Interestingly, the performance at high occupancies seems little affected by blocksize 32 versus 64.

For Volkov experiments, we want to be able to control the percentage occupancy (ie slide 30 et al).  Can we do that?  The next experiments try to vary the occupancy from 5% to 100%, assuming that we can fit 32 blocks per multicore.

<img src="img/occupancy_940m.png?raw=true" width="600" height="400" />

[results/occupancy_dyn_940m.tsv](results/occupancy_dyn_940m.tsv)

It looks like we get the maximum at around 16 blocks per multicore, rather than 32.  Why?  Actually, the peak is for 22 blocks per multicore, which is mysterious.

Here is the same graph for Titan X:

<img src="img/occupancy_titanx.png?raw=true" width="600" height="400" />

[results/occupancy_dyn_titanx.tsv](results/occupancy_dyn_titanx.tsv)

This graph looks more like what we'd expect, since it's upward floating over more or less the entire range.  However, the peak is at 24, not 32, which is strange again.  Mysterious!

## other things to check maybe

* Effect of barrier on performance
* Effect of memory layout for writes to global memory

## Reproduce Volkov's results

Reference: http://sbel.wisc.edu/Courses/ME964/Literature/talkVolkov10-GTC.pdf

These experiments are carried out on 940M, using opencl, except where otherwise stated.

### maths operation fma, vary ilp

[gpuexperiments/volkov1.py](gpuexperiments/volkov1.py)

[results/volkov1_940m.tsv](results/volkov1_940m.tsv)

[results/volkov1_titanx.tsv](results/volkov1_titanx.tsv)

<img src="img/volkov1_940m.png?raw=true" width="600" height="400" />

<img src="img/volkov1_titanx.png?raw=true" width="600" height="400" />

On 940M, with no ilp, we never actually hit the peak, using 1 block of maximum threads.  Seems like we hit some kind of other bottleneck, that is worked around by using ilp of 2 or more.

For Titan X, we get benefit of ilp all the way up to ilp==8.  For ilp 6 and 8, the unroll was reduced from 256 down to 64, in order to get the highest performance (just trying different unrolls empirically).  Using ilp 8, we get peak flops with a blocksize of only 128.  Note that the graph of flops versus blocksize is not contiguous, for ilp 6 and 8.

### global mem copy, ilp==1



## Hardware used

- 940M: Thinkpad T450s laptop
- Titan X: http://nimbix.net  I'm not affiliated with nimbix, nor do I receive sponsorship from them.  I think they provide a great service, and I hope that lots of people use them, so they can be profitable and survive :-)  Per-second billing, and Titan X is a breath of fresh air after using EC2 for many months

## Theoretical limits

940M, GM108M (rev a2):
- memory bandwidth: 14.40GB/s
- flops: 752 GFLOPS (`980MHz * 384 cuda-cores * 2 ops-per-fma / 1000`)
- compute units (==SMMs): 3  (from clinfo `max compute units`)
- clock frequency: 980MHz (from clinfo `max clock frequency`)
- shared memory: 48KiB  (from clinfo `Local memory size`)
- CUDA cores: 384 (from https://en.wikipedia.org/wiki/GeForce_900_series core config 'shader processors')

Titan X:
- memory bandwidth: 336GB/s
- flops: 6610 GFLOPS (`128 cuda-cores * 24 compute-units *2 ops-per-fma *1076MHz *1000*1000 mega /1000/1000/1000 giga`)
- clock: 1076MHz (from clinfo `max clock frequency`, on nimbix ngd3 instance)
- compute units (==SMMs): 24  (from clinfo `max compute units`)
- cuda cores per compute unit: 128
- L1 cache: 48KB
- shared memory: 96KB
- CUDA cores: 3072

References:
- https://en.wikipedia.org/wiki/GeForce_900_series
- https://www.techpowerup.com/gpudb/2643/geforce-940m
- http://www.tomshardware.com/reviews/nvidia-geforce-gtx-titan-x-gm200-maxwell,4091.html

Physical limits for SM5.0, corresponding to 940M, from http://developer.download.nvidia.com/compute/cuda/CUDA_Occupancy_calculator.xls :
```
Threads per Warp	32
Max Warps per Multiprocessor	64
Max Thread Blocks per Multiprocessor	32
Max Threads per Multiprocessor	2048
Maximum Thread Block Size	1024
Registers per Multiprocessor	65536
Max Registers per Thread Block	65536
Max Registers per Thread	255
Shared Memory per Multiprocessor (bytes)	65536
Max Shared Memory per Block	49152
Register allocation unit size	256
Register allocation granularity	warp
Shared Memory allocation unit size	256
Warp allocation granularity	4
```

For SM5.2, corresponding to Titan X, using the same calculator sheet:
```
Threads per Warp	32
Max Warps per Multiprocessor	64
Max Thread Blocks per Multiprocessor	32
Max Threads per Multiprocessor	2048
Maximum Thread Block Size	1024
Registers per Multiprocessor	65536
Max Registers per Thread Block	65536
Max Registers per Thread	255
Shared Memory per Multiprocessor (bytes)	98304
Max Shared Memory per Block	49152
Register allocation unit size	256
Register allocation granularity	warp
Shared Memory allocation unit size	256
Warp allocation granularity	4
```

- [clinfo output for 940M](results/clinfo_940.md)
- [clinfo output for Titan X](results/clinfo_titanx.md)

