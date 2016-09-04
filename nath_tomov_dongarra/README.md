# An Improved MAGMA GEMM for Fermi GPUs

Experiments based on "An Improved MAGMA GEMM for Fermi GPUs" paper, by Nath, Tomov, Dongarra

## GEMM for GTX280

### Results on 940M

- naive, but working, [gtx280.py](nath_tomov_dongarra/gtx280.py)
```
total time     0.65s per iteration 0.065s
to/from global 0.193 GB/s
to/from cores  65.9 GB/s
flops          32.9 GFLOPS/s
```

- remove initialization of C_row
```
```
