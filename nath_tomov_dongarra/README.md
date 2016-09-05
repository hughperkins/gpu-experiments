# An Improved MAGMA GEMM for Fermi GPUs

Experiments based on "An Improved MAGMA GEMM for Fermi GPUs" paper, by Nath, Tomov, Dongarra

## GEMM for GTX280

### Results on 940M

- naive, but working, [gtx280.py](gtx280.py), opencl at [gtx280.jinja2.cl](gtx280.jinja2.cl)
```
total time     1.19s; per iteration 0.060s
to/from global 0.211 GB/s
flops          36.0 GFLOPS/s
```

- remove initialization of C_row
```
total time     1.21s; per iteration 0.061s
to/from global 0.208 GB/s
flops          35.5 GFLOPS/s
```
(~1.5% change)
- add barriers before/after copying to shared memory
```
total time     1.23s; per iteration 0.061s
to/from global 0.205 GB/s
flops          35.0 GFLOPS/s
```
(~3% change)
- remove pull down B to shared memory
```
total time     1.14s; per iteration 0.057s
to/from global 0.221 GB/s
flops          37.7 GFLOPS/s
```
(~7.5% change)
- naive minus pull down A_row to registers
```
total time     0.94s; per iteration 0.047s
to/from global 0.269 GB/s
flops          45.9 GFLOPS/s
```
(28% change)
- naive minus pull down B_block, A_row
```
total time     0.90s; per iteration 0.045s
to/from global 0.280 GB/s
flops          47.9 GFLOPS/s
```
(33% change)
- naive minus pull down B_block, minus pull down A_row, minus calculate C_row
```
total time     0.02s; per iteration 0.001s
to/from global 11.379 GB/s
flops          1942.0 GFLOPS/s
```
memory bandwidth to/from global is about one third of peak (considering the calculation assumes we copied down A and B,
which we didnt do...)
- naive, no A down, no B down, no C calc; M=N=K=4096
```
total time     0.28s; per iteration 0.014s
to/from global 14.460 GB/s
flops          9871.5 GFLOPS/s
```
(doesnt change calculated bandwidht to/from global much)

- naive, no A down, no B down, no C calc; change C out loop termination conditoin to jinja2 variable, ie compile time
constant
```
total time     0.0028s; per iteration 0.000s
to/from global 89.172 GB/s
flops          15218.7 GFLOPS/s
```
Time is a bit quick.  Lets set M=N=K=4096
- naive, no A down, no B down, no C calc; change C out loop termination conditoin to jinja2 variable, ie compile time
constant, M=N=K=4096
```
total time     0.0032s; per iteration 0.000s
to/from global 1242.349 GB/s
flops          848110.6 GFLOPS/s
```
Seems time is independent of matrix size, and global bandwidth exceeds hardware capability.  Is the loop being optimized out?
Change it to write out a constant:
- naive, no A down, no B down, no C calc; change C out loop termination conditoin to jinja2 variable, ie compile time
constant, M=N=K=4096, write constant 123.0f to C
```
total time     0.2445s; per iteration 0.012s
to/from global 16.471 GB/s
flops          11244.2 GFLOPS/s
```
back in sensible realm, but still half of theoretical (since the global bandwidth assumes we are reading A, B; writing C;
but we are in fact simply writing C; so we should divide by 3)
Add #pragma unroll:
- naive, no A down, no B down, no C calc; change C out loop termination conditoin to jinja2 variable, ie compile time
constant, write constant 123.0f to C; M=N=K=4096; make C out loop #pragma unroll
```
total time     0.2442s; per iteration 0.012s
to/from global 16.486 GB/s
flops          11254.2 GFLOPS/s
```
either no change, or slower
Try change to float4
- naive, no A down, no B down, no C calc; change C out loop termination conditoin to jinja2 variable, ie compile time
constant, write constant 123.0f to C; M=N=K=4096; make C out loop #pragma unroll, float4 for c out
```
total time     0.0908s; per iteration 0.005s
to/from global 44.327 GB/s
flops          30260.4 GFLOPS/s
```
44.327 / 3 is 14.8GB/s , which is peak, so we are at peak for writing out C.

Let's add back in C calc:
- naive, no A down, no B down; change C out loop termination conditoin to jinja2 variable; make C out loop #pragma unroll, float4 for c out
```
total time     0.8552s; per iteration 0.043s
to/from global 0.294 GB/s
flops          50.2 GFLOPS/s
```
Ouch.
Use jinja2 constant for the C calc loop temrination conditions:
- naive, no A down, no B down; C out optimizations; C calc loop terminations are jinja2 vars
```
total time     0.2502s; per iteration 0.013s
to/from global 1.006 GB/s
flops          171.6 GFLOPS/s
```
Better.
Switch order of C calc loop, so blockCol is on inside:
- naive, no A down, no B down; C out optimizations; C calc loop terminations are jinja2 vars; blockCol C calc inner
```
total time     0.2338s; per iteration 0.012s
to/from global 1.077 GB/s
to/from cores  1102.5 GB/s
flops          183.7 GFLOPS/s
```
Not much difference, fractionally better.
We are well under peak bandwidth. We are three times under peak flops.  What if we replace B_block with constant?
- naive, no A down, no B down; C out optimizations; C calc loop terminations are jinja2 vars; blockCol C calc inner;
use contsnat instance of B_block in C calc
```
total time     0.0659s; per iteration 0.003s
to/from global 3.816 GB/s
to/from cores  3908.1 GB/s
flops          651.3 GFLOPS/s
```
approximately peak flops,  So reading from B_block is the limitation.
if we look at to/from cores bandwidth, from previous reading, ie 1102.5GB/s,
 its almost identical to three times the theoretical peak shared memory bandwidth of 940M, ie 375GB/seconds
So, that makes sense: we should be being limited by the shared memory bandwidth.  And to improve that, we'd need to
block over this, storing in registers.

So, at this point, lets save the updated gtx280.jinja2.cl as [gtx280_v3.jinja2.cl](gtx280_v3.jinja2.cl) , and rerun.  Results are:
```
total time     0.4250s; per iteration 0.021s
to/from global 0.592 GB/s
to/from cores  606.3 GB/s
flops          101.1 GFLOPS/s
```
We're three times faster than right at the start of the readme :-)  We're up to 13% of theoretical flops now.

We'll use gtx280_v3.jinja2.cl as the new baseline, from this point on in this readme.

Take out A and B reading again:
- gtx280_v3, no A read, no B read
```
total time     0.2347s; per iteration 0.012s
to/from global 1.072 GB/s
to/from cores  1097.9 GB/s
flops          183.0 GFLOPS/s
```
Put back in A read:
- gtx280_v3, no B read:
```
total time     0.3509s; per iteration 0.018s
to/from global 0.717 GB/s
to/from cores  734.4 GB/s
flops          122.4 GFLOPS/s
```
Try with no A read:
- gtx280_v3, no A read:
```
total time     0.2487s; per iteration 0.012s
to/from global 1.012 GB/s
to/from cores  1036.2 GB/s
flops          172.7 GFLOPS/s
```
Looks like A read is slowing it down the most.  Lets take out B read for now (to keep things simple(r),
and keep in A read, ie:
- gtx280_v3, no B read:
```
total time     0.3509s; per iteration 0.018s
to/from global 0.717 GB/s
to/from cores  734.4 GB/s
flops          122.4 GFLOPS/s
```
Presumably this sucks becuase the A read is uncoallesced, unlike B read which is coallesced okish.

To coallesce, we need to first read A into shared memory, I think?  (or, we could make the matrices column major?)
lets use shared memory for now, and think about transposing the matrix later, if that doesnt work
So, we will first read a block of A into shared memory, then read the data we want for our thread into private
memory

That didnt work, redcued occupancy too much it seems.

After modifications in [gtx280_v4.jinja2.cl](gtx280_v4.jinja2.cl), now get:
```
total time     0.2639s; per iteration 0.013s
to/from global 0.954 GB/s
to/from cores  976.6 GB/s
flops          162.8 GFLOPS/s
```

float4'ing the B block download:
```
total time     0.1953s; per iteration 0.010s
to/from global 1.289 GB/s
to/from cores  1319.6 GB/s
flops          219.9 GFLOPS/s
```
Lets call this [gtx280_v5.jinja2.cl](gtx280_v5.jinja2.cl)
