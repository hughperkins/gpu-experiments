# Basics 1

## Effect of optimization on performance?

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

## inlining?

Do functions get inlined?  When?  [gpuexperiments/inline.py](gpuexperiments/inline.py)

With optimizations off, it makes a difference (not shown here), but since the previous section demonstrated we should be testing with optimizations on anyway, let's check with optimizations on.  Here are the results on 940M:

<img src="img/inline_940m.png?raw=true" width="600" height="400" />

[results/inline_940m.tsv](results/inline_940m.tsv)

Meaning of labels:
- `void`: standard function, no inlining specified
- `define`: use a macro, `#define`, instead of a function
- `static inline`: use static inline function

They all run in the exact same time, so no need for `static inline` and so on.  At least, not in this case.

## maths

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

## Occupancy

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

