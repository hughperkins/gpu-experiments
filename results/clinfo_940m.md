```
Number of platforms                               1
  Platform Name                                   NVIDIA CUDA
  Platform Vendor                                 NVIDIA Corporation
  Platform Version                                OpenCL 1.2 CUDA 8.0.20
  Platform Profile                                FULL_PROFILE
  Platform Extensions                             cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp64 cl_khr_byte_addressable_store cl_khr_icd cl_khr_gl_sharing cl_nv_compiler_options cl_nv_device_attribute_query cl_nv_pragma_unroll cl_nv_copy_opts
  Platform Extensions function suffix             NV

  Platform Name                                   NVIDIA CUDA
Number of devices                                 1
  Device Name                                     GeForce 940M
  Device Vendor                                   NVIDIA Corporation
  Device Vendor ID                                0x10de
  Device Version                                  OpenCL 1.2 CUDA
  Driver Version                                  361.42
  Device OpenCL C Version                         OpenCL C 1.2 
  Device Type                                     GPU
  Device Profile                                  FULL_PROFILE
  Device Topology (NV)                            PCI-E, 04:00.0
  Max compute units                               3
  Max clock frequency                             980MHz
  Compute Capability (NV)                         5.0
  Device Partition                                (core)
    Max number of sub-devices                     1
    Supported partition types                     None
  Max work item dimensions                        3
  Max work item sizes                             1024x1024x64
  Max work group size                             1024
  Preferred work group size multiple              32
  Warp size (NV)                                  32
  Preferred / native vector sizes                 
    char                                                 1 / 1       
    short                                                1 / 1       
    int                                                  1 / 1       
    long                                                 1 / 1       
    half                                                 0 / 0        (n/a)
    float                                                1 / 1       
    double                                               1 / 1        (cl_khr_fp64)
  Half-precision Floating-point support           (n/a)
  Single-precision Floating-point support         (core)
    Denormals                                     Yes
    Infinity and NANs                             Yes
    Round to nearest                              Yes
    Round to zero                                 Yes
    Round to infinity                             Yes
    IEEE754-2008 fused multiply-add               Yes
    Support is emulated in software               No
    Correctly-rounded divide and sqrt operations  Yes
  Double-precision Floating-point support         (cl_khr_fp64)
    Denormals                                     Yes
    Infinity and NANs                             Yes
    Round to nearest                              Yes
    Round to zero                                 Yes
    Round to infinity                             Yes
    IEEE754-2008 fused multiply-add               Yes
    Support is emulated in software               No
    Correctly-rounded divide and sqrt operations  No
  Address bits                                    64, Little-Endian
  Global memory size                              1073610752 (1024MiB)
  Error Correction support                        No
  Max memory allocation                           268402688 (256MiB)
  Unified memory for Host and Device              No
  Integrated memory (NV)                          No
  Minimum alignment for any data type             128 bytes
  Alignment of base address                       4096 bits (512 bytes)
  Global Memory cache type                        Read/Write
  Global Memory cache size                        49152
  Global Memory cache line                        128 bytes
  Image support                                   Yes
    Max number of samplers per kernel             32
    Max size for 1D images from buffer            134217728 pixels
    Max 1D or 2D image array size                 2048 images
    Max 2D image size                             16384x16384 pixels
    Max 3D image size                             4096x4096x4096 pixels
    Max number of read image args                 256
    Max number of write image args                16
  Local memory type                               Local
  Local memory size                               49152 (48KiB)
  Registers per block (NV)                        65536
  Max constant buffer size                        65536 (64KiB)
  Max number of constant args                     9
  Max size of kernel argument                     4352 (4.25KiB)
  Queue properties                                
    Out-of-order execution                        Yes
    Profiling                                     Yes
  Prefer user sync for interop                    No
  Profiling timer resolution                      1000ns
  Execution capabilities                          
    Run OpenCL kernels                            Yes
    Run native kernels                            No
    Kernel execution timeout (NV)                 Yes
  Concurrent copy and kernel execution (NV)       Yes
    Number of async copy engines                  1
  printf() buffer size                            1048576 (1024KiB)
  Built-in kernels                                
  Device Available                                Yes
  Compiler Available                              Yes
  Linker Available                                Yes
  Device Extensions                               cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp64 cl_khr_byte_addressable_store cl_khr_icd cl_khr_gl_sharing cl_nv_compiler_options cl_nv_device_attribute_query cl_nv_pragma_unroll cl_nv_copy_opts

NULL platform behavior
  clGetPlatformInfo(NULL, CL_PLATFORM_NAME, ...)  No platform
  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, ...)   No platform
  clCreateContext(NULL, ...) [default]            No platform
  clCreateContext(NULL, ...) [other]              Success [NV]
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CPU)  No platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU)  No platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ACCELERATOR)  No platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CUSTOM)  No platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ALL)  No platform
```
