# Hardware and theoretical characteristics

## Hardware used

- 940M: Thinkpad T450s laptop
- Titan X: http://nimbix.net  I'm not affiliated with nimbix, nor do I receive sponsorship from them.  I think they provide a great service, and I hope that lots of people use them, so they can be profitable and survive :-)  Per-second billing, and Titan X is a breath of fresh air after using EC2 for many months

## Device physical characteristics

940M, GM108M (rev a2):
- architecture: Maxwell
- memory bandwidth: 14.40GB/s
- flops: 752 GFLOPS (`980MHz * 384 cuda-cores * 2 ops-per-fma / 1000`)
- compute units (==SMMs): 3  (from clinfo `max compute units`)
- clock frequency: 980MHz (from clinfo `max clock frequency`)
- Max shared memory per block: 48KiB  (from clinfo `Local memory size`)
- CUDA cores: 384 (from https://en.wikipedia.org/wiki/GeForce_900_series core config 'shader processors')
- shared memory bandwidth

Titan X:
- architecture: Maxwell
- memory bandwidth: 336GB/s
- flops: 6610 GFLOPS (`128 cuda-cores * 24 compute-units *2 ops-per-fma *1076MHz *1000*1000 mega /1000/1000/1000 giga`)
- clock: 1076MHz (from clinfo `max clock frequency`, on nimbix ngd3 instance)
- compute units (==SMMs): 24  (from clinfo `max compute units`)
- cuda cores per compute unit: 128
- L1 cache: 48KB
- Max shared memory per block: 48KiB  (from clinfo `Local memory size`)
- CUDA cores: 3072

Calculating shared memory bandwidth:
- reference [4], section 6.1 paragraph 1: theoretical peak SM bandwidth is:
```
f_core * W_bank * 32 * num-SMs
```
- for 940M, this is:
```
980 * 1000 * 1000 * 4 * 32 * 3
= 375 GB/second
```

References:
- [1] https://en.wikipedia.org/wiki/GeForce_900_series
- [2] https://www.techpowerup.com/gpudb/2643/geforce-940m
- [3] http://www.tomshardware.com/reviews/nvidia-geforce-gtx-titan-x-gm200-maxwell,4091.html
- [4] "Dissecting GPU Memory Hierarchy through Microbenchmarking", Mei, Chu 2016 https://arxiv.org/abs/1509.02308

## Compute capability limits

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

Limits for SM6.1, corresponding to GTX1080:
```
Shared memory per SM: 64KiB (http://www.hardware.fr/articles/948-2/gp104-7-2-milliards-transistors-16-nm.html "La mémoire partagée des SM du GP100 passe de 96 à 64 Ko mais elle n'est associée qu'à deux partitions au lieu de 4 ce qui indique en réalité une augmentation relative de 33%.")
```

## Clinfo output

- [clinfo output for 940M](results/clinfo_940m.md)
- [clinfo output for Titan X](results/clinfo_titanx.md)

