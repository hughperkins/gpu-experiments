# Old experiments

These experiments have been removed from main README, because they have issues that compromise their usage.  Some examples of issues:
- looks like building without optimizations is not representative of performance with optimizations.  The difference in runtime execution times is order(s) of magnitude

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

## maths

- we use shared memory, to avoid both things being optimized away (private memory), or global memory associated slowdowns (global memory)
- [gpuexperiments/old/calcs.py](gpuexperiments/old/calcs.py)

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

### shared memory

[gpuexperiments/old/sharedmemory.py](gpuexperiments/old/sharedmemory.py)

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

### for loops

[gpuexperiments/old/forloop.py](gpuexperiments/old/forloop.py)

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

[gpuexperiments/old/gridsize.py](gpuexperiments/old/gridsize.py)

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

[gpuexperiments/old/occupancy.py](gpuexperiments/old/occupancy.py)

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

[gpuexperiments/old/globalload.py](gpuexperiments/old/globalload.py)

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


