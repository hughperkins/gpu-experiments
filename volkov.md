# Volkov slide experiments

Reference: http://sbel.wisc.edu/Courses/ME964/Literature/talkVolkov10-GTC.pdf

These experiments are carried out on 940M, using opencl, except where otherwise stated.

## maths operation fma, vary ilp

[gpuexperiments/volkov1.py](gpuexperiments/volkov1.py)

<img src="img/volkov1_940m.png?raw=true" width="600" height="400" />

[results/volkov1_940m.tsv](results/volkov1_940m.tsv)

On 940M, with no ilp, we never actually hit the peak, using 1 block of maximum threads.  Seems like we hit some kind of other bottleneck, that is worked around by using ilp of 2 or more.  This is using unroll 64 throughout.  ilp 6 and 8 has a surprisingly low value for blocksize 192.

<img src="img/volkov1_titanx.png?raw=true" width="600" height="400" />

[results/volkov1_titanx.tsv](results/volkov1_titanx.tsv)

For Titan X, we get benefit of ilp all the way up to ilp==8.  For ilp 6 and 8, the unroll was reduced from 256 down to 64, in order to get the highest performance (just trying different unrolls empirically).  Using ilp 8, we get peak flops with a blocksize of only 128.  Note that the graph of flops versus blocksize does not increase monotonically, for ilp 6 and 8.

1080:

<img src="img/volkov1_1080.png?raw=true" width="600" height="400" />

[results/volkov1_1080.tsv](results/volkov1_1080.tsv)

Double the flops basically.

## global mem copy, vary ilp

[gpuexperiments/volkov_memcpy.py](gpuexperiments/volkov_memcpy.py)

<img src="img/volkov_memcpy_940m.png?raw=true" width="600" height="400" />

[results/volkov_memcpy_940m.tsv](results/volkov_memcpy_940m.tsv)

<img src="img/volkov_memcpy_titanx.png?raw=true" width="600" height="400" />

[results/volkov_memcpy_titanx.tsv](results/volkov_memcpy_titanx.tsv)

## Matrix multiplication, vary outputs per thread

[gpuexperiments/volkov_mm.py](gpuexperiments/volkov_mm.py)

<img src="img/volkov_mm_940m.png?raw=true" width="600" height="400" />

[results/volkov_mm_940m.tsv](results/volkov_mm_940m.tsv)

Drawing a graph with respect to matrix size is not strictly part of the Volkov slides, but I think it's a natural extension of how the earlier graphs work, and quite interesting.

We can see that larger matrices give higher flops.  But there is a sweet spot, somewhere around a matrix size of ~512.  This corresponds to ~1MB per input matrix, or about 2MB for the combined input matrices.  This sounds comparable in size to L2 cache perhaps?

For Titan X:

<img src="img/volkov_mm_titanx.png?raw=true" width="600" height="400" />

[results/volkov_mm_titanx.tsv](results/volkov_mm_titanx.tsv)

