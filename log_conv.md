#### Description:
1. cuda version of the `convolution_layer`
2. the most simple implementation, where 4d arrays were converted to 1-d array on the device 
3. invoked with <<<1,1>>>, thus this is the single thread benchmark

```
==4212== NVPROF is profiling process 4212, command: conv.exe
==4212== Profiling application: conv.exe
==4212== Warning: 7 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==4212== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   86.60%  34.1759s         1  34.1759s  34.1759s  34.1759s  convolution_layer_cuda(float*, float*, float*)
                   13.40%  5.28670s         2  2.64335s  13.792us  5.28669s  [CUDA memcpy HtoD]
                    0.01%  2.7689ms         1  2.7689ms  2.7689ms  2.7689ms  [CUDA memcpy DtoH]
      API calls:   99.39%  34.1761s         1  34.1761s  34.1761s  34.1761s  cudaDeviceSynchronize
                    0.47%  160.19ms         4  40.049ms  670.30us  158.11ms  cudaMalloc
                    0.12%  42.448ms         1  42.448ms  42.448ms  42.448ms  cuDevicePrimaryCtxRelease
                    0.02%  5.7384ms         3  1.9128ms  76.600us  2.8988ms  cudaMemcpy
                    0.00%  861.80us         1  861.80us  861.80us  861.80us  cudaLaunchKernel
                    0.00%  810.20us         4  202.55us  158.90us  304.40us  cudaFree
                    0.00%  34.800us         1  34.800us  34.800us  34.800us  cuModuleUnload
                    0.00%  15.000us         1  15.000us  15.000us  15.000us  cuDeviceTotalMem
                    0.00%  14.000us       101     138ns       0ns  1.4000us  cuDeviceGetAttribute
                    0.00%  4.1000us         3  1.3660us     200ns  3.3000us  cuDeviceGetCount
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuDeviceGetName
                    0.00%  1.4000us         2     700ns     200ns  1.2000us  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

```