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

