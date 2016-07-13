# Compound scenarios

These scenarios combine multiple components, eg global read, shared memory, private memory, global write.

## Matrix multiplication, batched

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

