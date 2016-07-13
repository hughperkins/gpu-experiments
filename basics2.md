# Basics 2

Continues from [basics1.md](basics1.md), but these experiments were done after the volkov reproducing experiments above, so placed into this section, after volkov.  Otherwise the comments wouldnt make much sense..

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

