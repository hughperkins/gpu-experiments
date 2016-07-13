# gpu-experiments

Informal experiments on various gpu kernel questions

With varying level of rigorousness...

Approximate target kernel I'm pondering as I write this https://github.com/hughperkins/neonCl-underconstruction/blob/c80492bd1fc5fd2e33ef3ad06f601a39a68ce9b3/winograd_kernels_cl.py#L317

Terminology will be interchangeably cuda/opencl.  Experiments will run on different devices.  Where not specified, they are running on NVIDIA 940M, which is a Maxwell.

## Basics

### Effect of optimization on performance?

[gpuexperiments/optimization2.py](gpuexperiments/optimization2.py)

<img src="img/optimization2_940m.png?raw=true" width="600" height="400" />

[results/optimization2_940m.tsv](results/optimization2_940m.tsv)

The labels mean:
- `opt`: optimizations on, else off
- `unroll`: manually unroll loops, using Jinja2, else `#pragma unroll`
- `fma`: use `fma(a,b,c)`, else `a*b+c`

Clearly optimization makes a huge difference
- even using manually unrolled, the unoptimized is still half as fast as optimized
- fma in non-optimized is actually slower than non-fma.  Looking at the ptx code, it looks like it's because `fma` is actually a function call, in non-optimized version, not written as an inline ptx assembler code
- `#pragma unroll` when non-optimized seems to be ignored

It is a mystery why even with manual Jinja2 unrolling, unoptimized is still slower than optimized.  The ptx code for these two versons is identical, to within register numbering.

Without any way of explaining this discrepancy, it looks like we should do all experiments with optimizations on, unfortunately.  This makes it harder to make artificial kernels for timing that dont just get entirely optimized away.

### inlining?

Do functions get inlined?  When?  [gpuexperiments/inline.py](gpuexperiments/inline.py)

With optimizations off, it makes a difference (not shown here), but since the previous section demonstrated we should be testing with optimizations on anyway, let's check with optimizations on.  Here are the results on 940M:

<img src="img/inline_940m.png?raw=true" width="600" height="400" />

[results/inline_940m.tsv](results/inline_940m.tsv)

Meaning of labels:
- `void`: standard function, no inlining specified
- `define`: use a macro, `#define`, instead of a function
- `static inline`: use static inline function

They all run in the exact same time, so no need for `static inline` and so on.  At least, not in this case.

### maths

[gpuexperiments/maths2.py](gpuexperiments/maths2.py)

Comparison of maths operators.  This is using a single block of 32 threads.  Results on 940M:

<img src="img/maths2_940m.png?raw=true" width="600" height="400" />

[results/maths2_940m.tsv](results/maths2_940m.tsv)

`int add` and `int sub` were optimized away initially, so needed some tricks to become measurable.

For floats, we see:
- `fma` is the fastest
- add/sub/mul all same speed as each other
- div about 3 times slower than mul

For ints:
- int mul half as fast as float mul
- int div slllooowwww
- float sqrt is pretty slow, but still faster than int div

On Titan X:

<img src="img/maths2_titanx.png?raw=true" width="600" height="400" />

[results/maths2_titanx.tsv](results/maths2_titanx.tsv)

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

<img src="img/volkov1_940m.png?raw=true" width="600" height="400" />

[results/volkov1_940m.tsv](results/volkov1_940m.tsv)

On 940M, with no ilp, we never actually hit the peak, using 1 block of maximum threads.  Seems like we hit some kind of other bottleneck, that is worked around by using ilp of 2 or more.  This is using unroll 64 throughout.  ilp 6 and 8 has a surprisingly low value for blocksize 192.

<img src="img/volkov1_titanx.png?raw=true" width="600" height="400" />

[results/volkov1_titanx.tsv](results/volkov1_titanx.tsv)

For Titan X, we get benefit of ilp all the way up to ilp==8.  For ilp 6 and 8, the unroll was reduced from 256 down to 64, in order to get the highest performance (just trying different unrolls empirically).  Using ilp 8, we get peak flops with a blocksize of only 128.  Note that the graph of flops versus blocksize does not increase monotonically, for ilp 6 and 8.

### global mem copy, vary ilp

[gpuexperiments/volkov_memcpy.py](gpuexperiments/volkov_memcpy.py)

<img src="img/volkov_memcpy_940m.png?raw=true" width="600" height="400" />

[results/volkov_memcpy_940m.tsv](results/volkov_memcpy_940m.tsv)

<img src="img/volkov_memcpy_titanx.png?raw=true" width="600" height="400" />

[results/volkov_memcpy_titanx.tsv](results/volkov_memcpy_titanx.tsv)

### Matrix multiplication, vary outputs per thread

[gpuexperiments/volkov_mm.py](gpuexperiments/volkov_mm.py)

<img src="img/volkov_mm_940m.png?raw=true" width="600" height="400" />

[results/volkov_mm_940m.tsv](results/volkov_mm_940m.tsv)

Drawing a graph with respect to matrix size is not strictly part of the Volkov slides, but I think it's a natural extension of how the earlier graphs work, and quite interesting.

We can see that larger matrices give higher flops.  But there is a sweet spot, somewhere around a matrix size of ~512.  This corresponds to ~1MB per input matrix, or about 2MB for the combined input matrices.  This sounds comparable in size to L2 cache perhaps?

For Titan X:

<img src="img/volkov_mm_titanx.png?raw=true" width="600" height="400" />

[results/volkov_mm_titanx.tsv](results/volkov_mm_titanx.tsv)

## Basics 2

Continues from `Basics` section earlier, but these experiments were done after the volkov reproducing experiments above, so placed into this section, after volkov.  Otherwise the comments wont make much sense..

## global write

[gpuexperiments/globalwrite.py](gpuexperiments/globalwrite.py)

Effect of ilp and blocks per SM, on Titan X:

<img src="img/globalwrite_varyilp_titanx.png?raw=true" width="600" height="400" />

[results/globalwrite_titanx.tsv](results/globalwrite_titanx.tsv)

ilp affects global writes, blocks per sm affects global writes.  Nothing too surprising here.  In line with volkov slides, and volkov results reproduction above.  What about, does stride affect global writes?  After all, dont have to wait for the global write to complete?  Just fire and forget?

<img src="img/globalwrite_varystride_titanx.png?raw=true" width="600" height="400" />

Interestingly, stride does make a difference.  And not a small one.  Presumably, writes to global memory somehow get coallesced, or involve cache lines, or something like this, so just writing one float is a lot less efficient than writing eg 32 adjacent floats.  Here is the same graph for 940M:

<img src="img/globalwrite_940m.png?raw=true" width="600" height="400" />

[results/globalwrite_940m.tsv](results/globalwrite_940m.tsv)

For 940M, the effect of stride 'maxes out' at a stride for 16.  For Titan X, this seems not to be the case: stride 32 is even slower than stride 16.  We should probably check also the result for stride 64, and maybe stride 128.

Moving on, here is effect of blocksize on bandwidth, given a single block, where the kernel contains a single tight for loop.

[gpuexperiments/globalwrite_blocksize.py](gpuexperiments/globalwrite_blocksize.py)

<img src="img/globalwrite_blocksize_940m.png?raw=true" width="600" height="400" />

[results/globalwrite_blocksize_940m.tsv](results/globalwrite_blocksize_940m.tsv)

We can see that bandwidth is linear in blocksize, until we saturate the available bandwidth, at a blocksize of ~128 threads.  Note that it looks like we can get full bandwidth using a single block, ie *running on a single compute unit*.

On Titan X:

<img src="img/globalwrite_blocksize_titanx.png?raw=true" width="600" height="400" />

[results/globalwrite_blocksize_titanx.tsv](results/globalwrite_blocksize_titanx.tsv)

Let's try varying similarly the gridsize, with a single block of 32 threads.

[gpuexperiments/globalwrite_gridsize.py](gpuexperiments/globalwrite_gridsize.py)

<img src="img/globalwrite_gridsize_940m.png?raw=true" width="600" height="400" />

[results/globalwrite_gridsize_940m.tsv](results/globalwrite_gridsize_940m.tsv)

A grid of 32 blocks of 32 threads is sufficient to saturate the bandwidth, on a 940M.  Actually, just 8 blocks of 32 threads reaches 12GiB/seconds, within about ~20% of full bandwidth.

On Titan X:

<img src="img/globalwrite_gridsize_titanx.png?raw=true" width="600" height="400" />

[results/globalwrite_gridsize_titanx.tsv](results/globalwrite_gridsize_titanx.tsv)

## Winograd-specific experiments

### Matrix multiplication, batched

This section was originally organized under the 'volkov' section above, since it is derived from the final matrix multiplication experiment.  It's not part of his slides though, and doesnt really fit into the story he was telling particularly well either.  It is part of my experiments into Winograd kernels, and belongs in this new section.  Prorbably.

[gpuexperiments/volkov_mm_batched.py](gpuexperiments/volkov_mm_batched.py)

<img src="img/volkov_mm_batched_940mb.png?raw=true" width="600" height="400" />

[results/volkov_mm_batched_940m.tsv](results/volkov_mm_batched_940m.tsv)

The flops are about 3 times lower than that of non-batched GEMM, even for large batch sizes.  For large matrix sizes, the number of outputs per thread makes little difference in flops.  At smaller bath sizes, increasing the outputs increases flops, but 32 outputs per thread is actually the slowest.

Note that the size of A and B are comparable, for batchsize 1024, versus matrix size 1024.  Compare:
```
A for matrix size 1024 = 1024 * 1024 ~= 1e6 floats
A for batch size 1024, with 32x32 matrices = 1024 * 32 *32 ~= 1e6 floats
```
Compare the ops in batched and non-batched
```
For non-batched, ops = 1024 * 1024 * 1024 * 2 ~= 2e9
For batched, ops = 1024 * 32 * 32 * 32 * 2 ~= 6e7
```
Therefore, the ratio of ops to memory transfer is about 33 times less for batched than non-batched, given equivalent input size, in number of floats, in both cases.

Let's calculate the maximum theoretical flops, given the physical limitations on global memory bandwidth, and assume that in the best case we will load the contents of A and B exactly once, and store C exactly once.

```
Total data transfer = (1024 * 32 * 32 * 4) * 3 /1024/1024/1024 GiB
= 0.0117GiB
Time to transfer 0.0117GiB = 0.0117 / 14.40 = 0.000813seconds
Therefore maximum flops = operations / 0.000813
= 1024 * 32 * 32 * 32 * 2 / 0.000813 / 1000/1000/1000 GFLOPS/second
= 6.71e7 / 0.000813/1e9 GFLOPS/second
= 82 GFLOPS/seconds
```
Looks like we have reached maximum theoretical flops for this scenario.  We should choose a more interesting scenario.

I think a more interesting scenario is something like:
- we have 100 A matrices
- 100 B matrices
- need to calcualte 100x100 C matrices, one for each possible A/B pair

In this case, maximum theoretical flops, based on bandwidth:
```
data transfer = (100 * 32 * 32 * 4) * 2 + 100*100*32*32*4 = 4.178e7bytes
time to transfer = 4.178e7 / 14.40e9 = 0.00290seconds
operations = 32*32*32*2*100*100 = 6.554e8 ops
therefore, maximum flops = 6.554e8 / 0.00290 / 1e9 gigaflops/second
= 226gigaflops/second
```
Still not ideal, but better.  It looks like the time to save the result matrices back up is not inconsiderable.

What about 1024 A matrices, and 1024 B matrices?
```
data transfer = (1024 * 32 * 32 * 4) * 2 + 1024*1024*32*32*4 = 4.303e9 bytes
time to transfer = 4.303e9 / 14.40e9 = 0.299seconds
operations = 32*32*32*2*1024*1024 = 6.872e10 operations
therefore, maximum flops = 6.872e10 / 0.299 / 1e9 gigaflops/second
= 229 gigaflops/second
```
Same

On Titan X:

<img src="img/volkov_mm_batched_titanx.png?raw=true" width="600" height="400" />

[results/volkov_mm_batched_titanx.tsv](results/volkov_mm_batched_titanx.tsv)

## Hardware and theoretical characteristics

### Hardware used

- 940M: Thinkpad T450s laptop
- Titan X: http://nimbix.net  I'm not affiliated with nimbix, nor do I receive sponsorship from them.  I think they provide a great service, and I hope that lots of people use them, so they can be profitable and survive :-)  Per-second billing, and Titan X is a breath of fresh air after using EC2 for many months

### Device physical characteristics

940M, GM108M (rev a2):
- memory bandwidth: 14.40GB/s
- flops: 752 GFLOPS (`980MHz * 384 cuda-cores * 2 ops-per-fma / 1000`)
- compute units (==SMMs): 3  (from clinfo `max compute units`)
- clock frequency: 980MHz (from clinfo `max clock frequency`)
- Max shared memory per block: 48KiB  (from clinfo `Local memory size`)
- CUDA cores: 384 (from https://en.wikipedia.org/wiki/GeForce_900_series core config 'shader processors')

Titan X:
- memory bandwidth: 336GB/s
- flops: 6610 GFLOPS (`128 cuda-cores * 24 compute-units *2 ops-per-fma *1076MHz *1000*1000 mega /1000/1000/1000 giga`)
- clock: 1076MHz (from clinfo `max clock frequency`, on nimbix ngd3 instance)
- compute units (==SMMs): 24  (from clinfo `max compute units`)
- cuda cores per compute unit: 128
- L1 cache: 48KB
- Max shared memory per block: 48KiB  (from clinfo `Local memory size`)
- CUDA cores: 3072

References:
- https://en.wikipedia.org/wiki/GeForce_900_series
- https://www.techpowerup.com/gpudb/2643/geforce-940m
- http://www.tomshardware.com/reviews/nvidia-geforce-gtx-titan-x-gm200-maxwell,4091.html

### Compute capability limits

From http://developer.download.nvidia.com/compute/cuda/CUDA_Occupancy_calculator.xls

Physical limits for SM5.0, corresponding to 940M:
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

Limits for SM5.2, corresponding to Titan X:
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

### Clinfo output

- [clinfo output for 940M](results/clinfo_940m.md)
- [clinfo output for Titan X](results/clinfo_titanx.md)

