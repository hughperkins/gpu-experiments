# gpu-experiments

## Specific

Informal experiments on various gpu kernel questions

With varying level of rigorousness...

Approximate target kernel I'm pondering as I write this https://github.com/hughperkins/neonCl-underconstruction/blob/c80492bd1fc5fd2e33ef3ad06f601a39a68ce9b3/winograd_kernels_cl.py#L317

Terminology will be interchangeably cuda/opencl.  Experiments will run on different devices.  Where not specified, they are running on NVIDIA 940M, which is a Maxwell.

I've moved most stuff out of this section into [old.md](old.md), after discovering that compiling without optimizations produces runtimes orders of magnitude different thant with optimization, even if the SASS is identical, see section 'Effect of optimization on performance?' below.

### Effect of barrier on performance


### Effect of workgroupsize on performance


### Effect of memory layout for writes to global memory


### If no optimization, does code with no side-effects get removed?

[gpuexperiments/optimization_shortcutting.py](gpuexperiments/optimization_shortcutting.py)

On 940M:
```
name		tot ms
kernel01	11.2
kernel02	12.0
kernel03	14.8
kernel04	12.5
kernel05	9.5
kernel06	9.4
kernel07	10.1
kernel08	11.5
kernel09	34.3
kernel10	23.0
kernel11	33.9
kernel12	32.3
kernel13	113.1
kernel14	115.4
kernel15	113.8
kernel16	48.4
kernel17	10.4
kernel18	13.4
kernel19	14.8
kernel20	26.3
kernel21	28.4
kernel22	36.0
kernel23	21.1
kernel24	25.0
kernel25	12.0
kernel26	17.8
kernel27	32.8
kernel28	84.4
kernel29	81.9
kernel30	84.3
kernel31	81.9
kernel32	85.8
```

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

### inlining?

Do functions get inlined?  When?  [gpuexperiments/inline.py](gpuexperiments/inline.py)

- with optimizations off, then:
  - no, see kernel 1, 2
  - #define runs 5 times faster!  (kernel 3)
- with optimizations on, #define and static inline run in same time (kernel 1,3,4,5)

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

### maths

[gpuexperiments/maths2.py](gpuexperiments/maths2.py)

Comparison of maths operators.  This is using a single block of 32 threads.  Results on 940M:
```
name		tot ms	op ns	gflops
float_add 	60.9	6.09	5.26
float_mul 	61.3	6.13	5.22
float_sub 	61.1	6.11	5.24
float_div 	185.5	18.55	1.72
int_mul   	133.1	13.31	2.40
int_div   	1485.0	148.50	0.22
int_add   	0.1	0.01	4386.14
int_sub   	0.1	0.01	4104.47
```
Clearly `int` add and sub are being optimized away here.  Not sure how to work around that?  Section 'effect of optimization on performance?' shows that just turning off optimizations gives unrepresentative results, so not really an option.

For other operations, we see:
- float add/sub/mul all same speed
- float div about 3 times slower
- int mul half as fast as float mul
- int div slllooowwww

## Reproduce Volkov's results

Reference: http://sbel.wisc.edu/Courses/ME964/Literature/talkVolkov10-GTC.pdf

These experiments are carried out on 940M, using opencl, except where otherwise stated.

### fma, ilp=1

[gpuexperiments/volkov1.py](gpuexperiments/volkov1.py)

```
kernel time(ms) GFLOPS
k1_nofma_128 24.447202682495117 41.88618277923521
k1_nofma_256 24.569988250732422 83.35372321307275
k1_nofma_384 24.61099624633789 124.82225299827564
k1_nofma_512 24.445295333862305 167.55780382518458
k1_nofma_640 31.03780746459961 164.9601056981764
k1_nofma_768 37.49728202819824 163.85187586075344
k1_nofma_896 44.58045959472656 160.7879341120096
k1_nofma_1024 49.462080001831055 165.62182584510674
k1_fma_128 24.608850479125977 41.611045622329655
k1_fma_256 24.442434310913086 83.78870835649977
k1_fma_384 25.36296844482422 121.12146914833615
k1_fma_512 24.651288986206055 166.15763996324776
k1_fma_640 30.616044998168945 167.2325736490854
k1_fma_768 36.604881286621094 167.84646702967459
k1_fma_896 44.046878814697266 162.73570779340284
k1_fma_1024 49.759864807128906 164.63067236521837
```

In theory we should get 752GFLOPS / 3 = 251 GLOPS, I think.  Seems like there is something wrong with my calculations somewhere?

Is it because of OpenCL?  Try with cuda, [gpuexperiments/volkov1_cuda.py](gpuexperiments/volkov1_cuda.py).  No change:
```
kernel time(ms) GFLOPS
k1_nofma_128 24.464130401611328 41.857200038982555
k1_nofma_256 24.43385124206543 83.81814146736532
k1_nofma_384 25.53868293762207 120.28811381946842
k1_nofma_512 24.553775787353516 166.8175206727128
k1_nofma_640 33.35285186767578 153.5101112286621
k1_nofma_768 37.592172622680664 163.43827907124236
k1_nofma_896 45.48525810241699 157.58952018828066
k1_nofma_1024 50.3230094909668 162.78835631780925
k1_fma_128 25.36296844482422 40.373823049445384
k1_fma_256 25.983810424804688 78.81830903618881
k1_fma_384 26.196956634521484 117.26552984218861
k1_fma_512 26.06511116027832 157.14492736336612
k1_fma_640 31.43763542175293 162.86212150858114
k1_fma_768 37.93835639953613 161.94692082325219
k1_fma_896 44.34680938720703 161.6350781273521
k1_fma_1024 49.98207092285156 163.89877107422248
```
So, that's not the reason...

On Titan X, OpenCL:
```
kernel time(ms) GFLOPS
k1_nofma_128 24.09195899963379 42.503808013933835
k1_nofma_256 24.11174774169922 84.93784946406676
k1_nofma_384 22.299766540527344 137.7592897404097
k1_nofma_512 22.337913513183594 183.3653800111002
k1_nofma_640 22.498369216918945 227.57204980660202
k1_nofma_768 22.606372833251953 271.7817690311966
k1_nofma_896 26.972532272338867 265.7518370031203
k1_nofma_1024 30.727624893188477 266.6004947820082
k1_fma_128 20.7827091217041 49.27172843556768
k1_fma_256 20.7827091217041 98.54345687113536
k1_fma_384 20.824670791625977 147.51733800446505
k1_fma_512 20.870208740234375 196.26061487844999
k1_fma_640 20.900726318359375 244.96756342398248
k1_fma_768 21.07858657836914 291.4806444519851
k1_fma_896 25.182008743286133 284.64766544531864
k1_fma_1024 28.693199157714844 285.50319380463327
```

Theoretical for titan x should be `6610/24 = 275 GFLOPS`, so this looks fairly ok...  a little high perhaps???

Titan X, cuda:
```
kernel time(ms) GFLOPS
k1_nofma_128 24.17612075805664 42.35584402674503
k1_nofma_256 24.148941040039062 84.80703135613301
k1_nofma_384 24.199724197387695 126.94359551137427
k1_nofma_512 21.901607513427734 187.01823587555242
k1_nofma_640 22.090911865234375 231.7695182179244
k1_nofma_768 21.797895431518555 281.8620733043849
k1_nofma_896 25.182008743286133 284.64766544531864
k1_nofma_1024 28.693437576293945 285.5008215107728
k1_fma_128 20.860910415649414 49.08702350937747
k1_fma_256 20.8437442779541 98.25489953674578
k1_fma_384 20.842552185058594 147.39077886067258
k1_fma_512 20.87712287902832 196.19561678752925
k1_fma_640 20.903587341308594 244.93403531182994
k1_fma_768 21.091222763061523 291.3060124119689
k1_fma_896 25.19679069519043 284.4806740156884
k1_fma_1024 28.69701385498047 285.4652418331063
```

... as for 940M, identical to OpenCL results.  At least, for fma.  For non-fma, it matches the fma speed of both OpenCL and CUDA, whereas the OpenCL version is slower, presumably not being optimized automtaiclaly into fma.

Interestingly, a single compute unit on the Titan X is not even twice as fast as one on the 940M.  Admittedly there are 8 times as many of them :-P

For the peak, peak is at threads = 768

### fma, ilp=2

940M:
```
k1_fma_ilp2_128 12.473583221435547 82.09349164723422
k1_fma_ilp2_256 12.477636337280273 164.1336503678227
k1_fma_ilp2_384 14.468669891357422 212.32082997726
k1_fma_ilp2_512 17.87877082824707 229.09852357012363
k1_fma_ilp2_640 20.66779136657715 247.72845386274759
k1_fma_ilp2_768 24.718523025512695 248.55854023553923
k1_fma_ilp2_896 30.41863441467285 235.6450293686562
k1_fma_ilp2_1024 33.705711364746094 243.04486297145118
```

Suddenly, we our hitting theoretical flops, interestingly.  We get these at threads=640.  Compared to peak at ilp=1 was threads=768 (but we never really hit the peak, for some reason).

Titan X:
```
kernel time(ms) GFLOPS
k1_fma_ilp2_128 12.244939804077148 83.62102879910155
k1_fma_ilp2_256 12.266874313354492 166.943010557348
k1_fma_ilp2_384 13.495922088623047 227.60974550894193
k1_fma_ilp2_512 16.16382598876953 253.38913316968885
k1_fma_ilp2_640 20.134925842285156 254.26824812278312
k1_fma_ilp2_768 24.16253089904785 254.26172488586843
k1_fma_ilp2_896 28.171539306640625 254.42490628513363
k1_fma_ilp2_1024 32.1955680847168 254.4286744823268
```
peak at threads=512

## fma, ilp=3

940M:
```
kernel time(ms) GFLOPS
k1_fma_ilp3_128 9.108543395996094 112.42192691864727
k1_fma_ilp3_256 9.561538696289062 214.19146698583683
k1_fma_ilp3_384 13.108015060424805 234.3604264901144
k1_fma_ilp3_512 17.731428146362305 231.00226141910153
k1_fma_ilp3_640 20.359277725219727 251.48239879147002
k1_fma_ilp3_768 24.341106414794922 252.4125195995847
k1_fma_ilp3_896 28.972864151000977 247.40391431933574
k1_fma_ilp3_1024 33.41960906982422 245.1255483834147
```

We hit the peak at threads=640, as for ilp=2.

Titan X:
```
kernel time(ms) GFLOPS
k1_fma_ilp3_128 9.255647659301758 110.6280729010859
k1_fma_ilp3_256 9.250879287719727 221.37019242251776
k1_fma_ilp3_384 12.419700622558594 247.33312704981893
k1_fma_ilp3_512 16.224384307861328 252.44334566307455
k1_fma_ilp3_640 20.10059356689453 254.70254412944536
k1_fma_ilp3_768 24.056196212768555 255.3856282872807
k1_fma_ilp3_896 25.90012550354004 276.7377033373965
k1_fma_ilp3_1024 27.630329132080078 296.46681633224995
```
Peak at around threads=384

### fma, ilp=4

Titan X
```
kernel time(ms) GFLOPS
k1_fma_ilp4_128 15.054941177368164 68.01318264459665
k1_fma_ilp4_256 11.542797088623047 177.41531036861468
k1_fma_ilp4_384 12.303590774536133 249.6672270958079
k1_fma_ilp4_512 16.300678253173828 251.2618059437213
k1_fma_ilp4_640 20.272493362426758 252.54280410760188
k1_fma_ilp4_768 24.258136749267578 253.25963191239296
k1_fma_ilp4_896 25.95353126525879 276.16824757848724
k1_fma_ilp4_1024 27.693510055541992 295.79044677150745
```
Peak still at threads=384

But, if reduce unroll from 256 to 128:
```
kernel time(ms) GFLOPS
k1_fma_ilp4_128 6.186723709106445 165.50512228190127
k1_fma_ilp4_256 8.220434188842773 249.11931425465102
k1_fma_ilp4_384 12.155532836914062 252.70824678878014
k1_fma_ilp4_512 16.148090362548828 253.6360500867005
k1_fma_ilp4_640 20.14446258544922 254.14787305857692
k1_fma_ilp4_768 24.261474609375 253.2247888026566
k1_fma_ilp4_896 25.955915451049805 276.1428800890204
k1_fma_ilp4_1024 27.69303321838379 295.79553988915006
```
Peak at 256.

### fma, ilp=8

Titan X, and concentrating on range up to 384 threads, 32 at a time:

With unroll 64:
```
kernel time(ms) GFLOPS
k1_fma_ilp8_32 3.6377906799316406 70.36787944181832
k1_fma_ilp8_64 3.6275386810302734 141.13350043026807
k1_fma_ilp8_96 3.621339797973633 212.06263174466994
k1_fma_ilp8_128 3.6163330078125 283.1416414882025
k1_fma_ilp8_160 7.024526596069336 182.2070231346543
k1_fma_ilp8_192 7.032871246337891 218.38899678417465
k1_fma_ilp8_224 6.975412368774414 256.88593265416307
k1_fma_ilp8_256 7.031917572021484 291.2248198340717
k1_fma_ilp8_288 10.41102409362793 221.2897139878942
k1_fma_ilp8_320 10.415315628051758 245.7761484544498
k1_fma_ilp8_352 10.425806045532227 270.0817340839238
k1_fma_ilp8_384 10.4217529296875 294.74920512168666
```

Seems like non-linear, but peak is at 128.  Reducing unroll to 32 drops the flops back down a bit:
```
kernel time(ms) GFLOPS
k1_fma_ilp8_32 4.360437393188477 58.70594917837301
k1_fma_ilp8_64 4.347801208496094 117.75313714885544
k1_fma_ilp8_96 4.351377487182617 176.4845385770529
k1_fma_ilp8_128 4.347562789916992 235.5191893662219
k1_fma_ilp8_160 8.20159912109375 156.05713728535815
k1_fma_ilp8_192 8.214235305786133 186.9804843450377
k1_fma_ilp8_224 8.311986923217773 215.57845657754203
k1_fma_ilp8_256 8.227348327636719 248.9099581600241
k1_fma_ilp8_288 12.206792831420898 188.73528664054723
k1_fma_ilp8_320 12.185811996459961 210.06693363918998
k1_fma_ilp8_352 12.176036834716797 231.2591374536108
k1_fma_ilp8_384 12.198448181152344 251.81919424359253
```
Peak is still more or less at 128 though.

Try ilp 6:

### fma, ilp=6

With unroll 64:
```
kernel time(ms) GFLOPS
k1_fma_ilp6_32 4.242181777954102 60.34244391183409
k1_fma_ilp6_64 4.234075546264648 120.9159417223114
k1_fma_ilp6_96 4.230022430419922 181.54770113683867
k1_fma_ilp6_128 4.229545593261719 242.09089166251724
k1_fma_ilp6_160 8.107662200927734 157.86524503370936
k1_fma_ilp6_192 8.10694694519043 189.4550077093075
k1_fma_ilp6_224 8.103132247924805 221.13489662702938
k1_fma_ilp6_256 8.083820343017578 253.3293469057427
k1_fma_ilp6_288 12.10474967956543 190.32632685409735
k1_fma_ilp6_320 12.092113494873047 211.69468522482433
k1_fma_ilp6_352 12.088775634765625 232.92845041329883
k1_fma_ilp6_384 12.089967727661133 254.0787089920757
```
Peak at 128.  Not quite the peak, but close.

## Hardware used

- 940M: Thinkpad T450s laptop
- Titan X: http://nimbix.net

## Theoretical limits

940M, GM108M (rev a2):
- memory bandwidth: 14.40GB/s
- flops: 752 GFLOPS (`980MHz * 384 cuda-cores * 2 ops-per-fma / 1000`)
- compute units (==SMMs): 3  (from clinfo `max compute units`)
- clock frequency: 980MHz (from clinfo `max clock frequency`)
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

