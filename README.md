# gpu-experiments

## Specific

Informal experiments on various gpu kernel questions

With varying level of rigorousness...

Approximate target kernel I'm pondering as I write this https://github.com/hughperkins/neonCl-underconstruction/blob/c80492bd1fc5fd2e33ef3ad06f601a39a68ce9b3/winograd_kernels_cl.py#L317

Terminology will be interchangeably cuda/opencl.  Experiments will run on different devices.  Where not specified, they are running on NVIDIA 940M, which is a Maxwell.

### Effect of barrier on performance



### Effect of workgroupsize on performance


### Effect of memory layout for writes to global memory


### If no optimization, does code with no side-effects get removed?

[gpuexperiments/optimization_shortcutting.py](gpuexperiments/optimization_shortcutting.py)

Observations:
- even with optimization off, if you dont save the value of a variable to global memory, or use it in some
other way, that variable entirely vanishes  (kernels 1 to 8 or so)
- it's enough to save each variable to the same location in global memory, in order for all variables to not be pruned (eg kernel 8)
- a float is stored as an unsigned integer, u32, at least, in the ptx (kernel 4 etc)
- even if you use integer variables, the values eg 1,2,3, are stored in float representation, if you're going to ultimately store in a float global array (eg kernel 4)
- you can freely change the global memory array between float and int.  This works ok (though obviously returned data values will be different...).  This affects how numbers stored to that array are represented in the ptx
- calling `get_global_id(0)` is a masssssiiiivvveee amount of ptx code (see kernel9)
- calling `get_local_id(0)` does too... (kernel10)
- changed to shared memory are not pruned, even if shared memory is never read (kernel17)
- variables containing constants, such variables with values added, such variables with similar variables added, all become the actual constant value (kernel19) (even though optimization is off)
- for-loops that simply add a constant to a variable a constant number of times are not pruned (this is with optimization off) (kernel20)
- said for loop has a noticeable effect on execution time (kernel20 vs kernel19)
- calls to get_local_id(0) are not cached (26 and 27), and adidtional calls take significant time
- get_global_id is slower than get_local_id (kernel 21, 23)
- get local id takes noticeable time (kernel 23, kernel 25)
- access pattern for writes to global memory seems to make little difference? (kernels 28 to 32)

[gpuexperiments/optimization2.py](gpuexperiments/optimization2.py)

```
k1_noopt_128 140.65217971801758 0.8474924483589801
k1_opt_128 9.615182876586914 12.39723276054452
k1_noprag_noopt_128 16.440153121948242 7.250641722862736
k1_noprag_opt_128 6.257295608520508 19.050028576871787
k1_fma_noopt_128 483.37507247924805 0.2466028286168346
k1_fma_opt_128 9.495258331298828 12.553809069452117
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
k1_opt_128 9.421348571777344 12.652292742180382
k1_noprag4_noopt_128 46.595096588134766 2.558244727120153
k1_noprag4b_noopt_128 48.75659942626953 2.444831295843521
```

### inlining?

Do functions get inlined?  When?  [gpuexperiments/inline.py](gpuexperiments/inline.py)

- no, see kernel 1, 2
- #define runs 5 times faster!  (kernel 3)
- with optimizatoins on, #define and static inline run in same time (kernel 1,3,4,5)

### shared memory

[gpuexperiments/sharedmemory.py](gpuexperiments/sharedmemory.py)

- with optimizations turned off, store to shared memory not optimized away (kernel01)

```
kernel_copy_local_from_global 34.26074981689453
kernel_copy_local_from_global_gid 48.19059371948242
kernel_copy_local_to_global 32.5167179107666
kernel_copy_local_to_global_gid 58.62689018249512
kernel_init_local 384.2778205871582
kernel_init_local_noloop 19.541501998901367
kernel_store_to_local 9.908437728881836
```

Titan X:
```
kernel_copy_local_from_global 4.500865936279297
kernel_copy_local_from_global_gid 3.6003589630126953
kernel_copy_local_to_global 4.492759704589844
kernel_copy_local_to_global_gid 3.99017333984375
kernel_init_local 28.486013412475586
kernel_init_local_noloop 1.9900798797607422
kernel_store_to_local 1.9931793212890625
```

### maths

- we use shared memory, to avoid both things being optimized away (private memory), or global memory associated slowdowns (global memory)
- [gpuexperiments/calcs.py](gpuexperiments/calcs.py)

```
kernel_float_add 10.192394256591797
kernel_float_div 17.50779151916504
kernel_float_mul 9.238243103027344
kernel_float_nop 9.191274642944336
kernel_int_add 10.240554809570312
kernel_int_div 51.10955238342285
kernel_int_mul 15.785455703735352
kernel_int_nop 9.370088577270508
kernel_int_shift 10.795354843139648
```
- int is much slower than float calcs
- int div is particularly slow
- float div is slower than mult, but not nearly as slow as int div
- float mul takes same time as nop :-O
- it seems like maybe int stuff happens via an int unit, that is particularly slow?
- float add is slower than float mul :-O

Titan X:
```
kernel01 2.2470951080322266
kernel_float_add 2.2232532501220703
kernel_float_div 2.227783203125
kernel_float_mul 2.229928970336914
kernel_float_nop 2.2292137145996094
kernel_int_add 2.2275447845458984
kernel_int_div 5.436897277832031
kernel_int_mul 2.2263526916503906
kernel_int_nop 2.223968505859375
kernel_int_shift 2.2313594818115234
```

### for loops

[gpuexperiments/forloop.py](gpuexperiments/forloop.py)

```
kernel_for_loop_1e4 1.1463165283203125
kernel_for_loop_1e5 13.753890991210938
kernel_for_loop_1e5_div_const 115.87834358215332
kernel_for_loop_1e5_float_add_const 14.91856575012207
kernel_for_loop_1e5_float_div_const 56.73623085021973
kernel_for_loop_1e5_float_mul_const 16.42632484436035
kernel_for_loop_1e5_mul_const 21.55447006225586
kernel_for_loop_1e5_sum 15.58828353881836
kernel_for_loop_1e5_sum_const 14.525890350341797
kernel_for_loop_1e6 109.52186584472656
```
Titan X:
```
kernel_for_loop_1e4 1.0766983032226562
kernel_for_loop_1e5 10.216712951660156
kernel_for_loop_1e5_div_const 97.53608703613281
kernel_for_loop_1e5_float_add_const 11.809587478637695
kernel_for_loop_1e5_float_div_const 47.07169532775879
kernel_for_loop_1e5_float_mul_const 13.068437576293945
kernel_for_loop_1e5_mul_const 17.682790756225586
kernel_for_loop_1e5_sum 11.813640594482422
kernel_for_loop_1e5_sum_const 11.833667755126953
kernel_for_loop_1e6 90.73615074157715
```

### grid size

[gpuexperiments/gridsize.py](gpuexperiments/gridsize.py)

940M:
```
grid_001 108.79731178283691
grid_008 109.05265808105469
grid_016 115.04936218261719
grid_032 118.40677261352539
grid_048 114.66002464294434
grid_064 172.91593551635742
grid_128 349.22146797180176
```

Titan X:
```
grid_001 92.00358390808105
grid_008 92.437744140625
grid_016 92.44227409362793
grid_032 92.4527645111084
grid_048 92.45562553405762
grid_064 92.45038032531738
grid_128 92.88883209228516
grid_256 93.48249435424805
grid_512 96.0536003112793
grid_768 129.60028648376465
grid_1024 237.98060417175293
```

### occupancy

- shared occupancy was fairly straightforward, but note that we start seeing an effect from as low as 8KB of shared memory
upwards
- (not shown here)  Changing the workgroup size from 32 to 512, meant that 8KB of shared memory was ok, but that
was presumably because the raiot of shared-memory/thread was lower
- for private memory, it took a lot of effort to get this working
    - registers are automtaically optimized away to hardly any, when going from OpenCL => PTX
    - this can be handled by eg storing/loading each register to/from shared memory
    - however, it turned out that even if the ptx is using eg 64 registers, the sass might not be
    - finally, a non-inlined function call was added, with the appropriate number of parameters correpsonding to the number of desired registers, and this conveted to a SASS file with the appropriat enumber of registers
- we can see that up to use of 64 registers, performance is unaffected by adding registers, then starts to rise

[gpuexperiments/occupancy.py](gpuexperiments/occupancy.py)

```
shared_0 34.07549858093262
shared_1 30.35902976989746
shared_2 32.41872787475586
shared_4 34.063100814819336
shared_8 67.08264350891113
shared_16 144.55866813659668
shared_32 422.792911529541
private_0 0.05435943603515625
private_1 0.07200241088867188
private_2 0.055789947509765625
private_4 0.08249282836914062
private_8 0.10442733764648438
private_16 0.11348724365234375
private_32 0.1761913299560547
private_64 0.32019615173339844
private_128 1.0952949523925781
private_256 3.1604766845703125
private_512 10.724306106567383
private_1024 36.50164604187012
private_2048 76.1110782623291
private_looped_0 104.78925704956055
private_looped_1 104.3093204498291
private_looped_2 104.57110404968262
private_looped_4 103.02019119262695
private_looped_8 104.13885116577148
private_looped_16 105.1950454711914
private_looped_32 105.42106628417969
private_looped_64 105.19862174987793
private_looped_128 193.23396682739258
private_looped_256 288.72108459472656
private_looped_512 288.3191108703613
private_looped_1024 321.1710453033447
private_looped_2048 355.73697090148926
```

### load from global

[gpuexperiments/globalload.py](gpuexperiments/globalload.py)

```
kernel_ilp_load1 35.62347094217936
kernel_ilp_load1b 37.770748138427734
kernel_ilp_load1c 41.27724965413412
kernel_ilp_load1d 33.61137708028158
kernel_ilp_load1e 15.196800231933594
kernel_ilp_load1f 15.071233113606771
kernel_ilp_load2 49.576759338378906
kernel_ilp_load2b 43.80480448404948
kernel_ilp_load3 46.45133018493652
kernel_ilp_load4 42.43572552998861
kernel_ilp_load5 41.69940948486328
```

## Reproduce Volkov's results

Reference: http://sbel.wisc.edu/Courses/ME964/Literature/talkVolkov10-GTC.pdf

These experiments are carried out on 940M, using opencl, except where otherwise stated.

[gpuexperiments/volkov1.py](gpuexperiments/volkov1.py)

```
k1_nofma_128 9.495258331298828 12.553809069452116
k1_nofma_256 9.752750396728516 24.44472693492397
k1_nofma_384 11.963844299316406 29.89047429254683
k1_nofma_512 12.195110321044922 39.09818181818182
k1_nofma_640 13.562679290771484 43.94473156839996
k1_nofma_768 15.676736831665039 45.62237464450291
k1_nofma_896 18.646955490112305 44.74787433992661
k1_nofma_1024 21.175622940063477 45.033540876183615
k1_fma_128 9.488821029663086 12.56232568657504
k1_fma_256 10.361433029174805 23.00872086334246
k1_fma_384 11.456012725830078 31.21548387096774
k1_fma_512 11.046409606933594 43.163947163947164
k1_fma_640 14.116287231445312 42.22132144304824
k1_fma_768 15.717744827270508 45.50334470989761
k1_fma_896 17.66180992126465 47.24383428502004
k1_fma_1024 21.21901512145996 44.94144878032338
```

In theory we should get 823.3GFLOPS / 3, I think.  Seems like there is something wrong with my calculations somewhere?

## Theoretical limits

940M, GM108M (rev a2):
- memory bandwidth: 14.40GB/s
- flops: 823.3GFLOPS (790.3 per wikipedia?)
- compute units (==SMMs): 3

Titan X:
- memory bandwidth: 336GB/s
- flops: 6144 GFLOPS
- compute units (==SMMs): 24
- L1 cache: 48KB
- shared memory: 96KB
- CUDA cores: 3072

References:
- https://en.wikipedia.org/wiki/GeForce_900_series
- https://www.techpowerup.com/gpudb/2643/geforce-940m
- http://www.tomshardware.com/reviews/nvidia-geforce-gtx-titan-x-gm200-maxwell,4091.html

