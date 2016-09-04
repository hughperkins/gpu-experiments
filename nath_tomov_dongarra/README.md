# An Improved MAGMA GEMM for Fermi GPUs

Experiments based on "An Improved MAGMA GEMM for Fermi GPUs" paper, by Nath, Tomov, Dongarra

## GEMM for GTX280

### Results on 940M

- naive, but working, [gtx280.py](nath_tomov_dongarra/gtx280.py)
```
total time     1.19s; per iteration 0.060s
to/from global 0.211 GB/s
to/from cores  72.1 GB/s
flops          36.0 GFLOPS/s
```

- remove initialization of C_row
```
total time     1.21s; per iteration 0.061s
to/from global 0.208 GB/s
to/from cores  71.0 GB/s
flops          35.5 GFLOPS/s
```
(~1.5% change)
- add barriers before/after copying to shared memory
```
total time     1.23s; per iteration 0.061s
to/from global 0.205 GB/s
to/from cores  70.0 GB/s
flops          35.0 GFLOPS/s
```
(~3% change)
- remove pull down B to shared memory
```
total time     1.14s; per iteration 0.057s
to/from global 0.221 GB/s
to/from cores  75.3 GB/s
flops          37.7 GFLOPS/s
```
(~7.5% change)
- naive minus pull down A_row to registers
```
total time     0.94s; per iteration 0.047s
to/from global 0.269 GB/s
to/from cores  91.8 GB/s
flops          45.9 GFLOPS/s
```
(28% change)
- naive minus pull down B_block, A_row
```
total time     0.90s; per iteration 0.045s
to/from global 0.280 GB/s
to/from cores  95.7 GB/s
flops          47.9 GFLOPS/s
```
(33% change)
- naive minus pull down B_block, minus pull down A_row, minus calculate C_row
```
total time     0.02s; per iteration 0.001s
to/from global 11.379 GB/s
to/from cores  3883.9 GB/s
flops          1942.0 GFLOPS/s
```
memory bandwidth to/from global is about one third of peak (considering the calculation assumes we copied down A and B,
which we didnt do...)
- naive, no A down, no B down, no C calc; M=N=K=4096
```
total time     0.28s; per iteration 0.014s
to/from global 14.460 GB/s
to/from cores  19742.9 GB/s
flops          9871.5 GFLOPS/s
```
(doesnt change calculated bandwidht to/from global much)

