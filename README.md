# gpu-experiments
Informal experiments on various gpu kernel questions

With varying level of rigorousness...

Approximate target kernel I'm pondering as I write this https://github.com/hughperkins/neonCl-underconstruction/blob/c80492bd1fc5fd2e33ef3ad06f601a39a68ce9b3/winograd_kernels_cl.py#L317

Terminology will be interchangeably cuda/opencl.  Experiments will run on different devices.  Where not specified, they are running on NVIDIA 940M, which is a Maxwell.

## Effect of barrier on performance



## Effect of workgroupsize on performance


## Effect of memory layout for writes to global memory


## If no optimization, does code with no side-effects get removed?

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

