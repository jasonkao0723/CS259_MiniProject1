#### Description 1:
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

#### Description 2:
1. cuda version of the `convultion_layer`
2. picking up threads = 32, invoked with <<<1,32>>>
3. tremendous speed up from 34.1759s to 872.88ms for `convolution_layer_cuda(float*, float*, float*)`
4. now the new bottleneck is the CUDA memcpy part (host to device)
```
==15968== NVPROF is profiling process 15968, command: ./conv.exe
==15968== Profiling application: ./conv.exe
==15968== Warning: 4 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==15968== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   85.78%  5.29497s         2  2.64748s  12.736us  5.29496s  [CUDA memcpy HtoD]
                   14.14%  872.88ms         1  872.88ms  872.88ms  872.88ms  convolution_layer_cuda(float*, float*, float*)
                    0.08%  4.7811ms         1  4.7811ms  4.7811ms  4.7811ms  [CUDA memcpy DtoH]
      API calls:   80.14%  873.11ms         1  873.11ms  873.11ms  873.11ms  cudaDeviceSynchronize
                   14.89%  162.22ms         4  40.555ms  656.40us  160.20ms  cudaMalloc
                    4.07%  44.330ms         1  44.330ms  44.330ms  44.330ms  cuDevicePrimaryCtxRelease
                    0.79%  8.6532ms         3  2.8844ms  69.000us  4.9783ms  cudaMemcpy
                    0.09%  1.0079ms         4  251.98us  182.40us  421.90us  cudaFree
                    0.01%  56.900us         1  56.900us  56.900us  56.900us  cudaLaunchKernel
                    0.00%  29.300us         1  29.300us  29.300us  29.300us  cuModuleUnload
                    0.00%  12.500us       101     123ns       0ns  1.0000us  cuDeviceGetAttribute
                    0.00%  10.200us         1  10.200us  10.200us  10.200us  cuDeviceTotalMem
                    0.00%  4.7000us         3  1.5660us     200ns  3.9000us  cuDeviceGetCount
                    0.00%  1.8000us         2     900ns     200ns  1.6000us  cuDeviceGet
                    0.00%  1.6000us         1  1.6000us  1.6000us  1.6000us  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```

#### Description 3:
1. cuda version of the `convultion_layer`
2. picking up threads = 256, invoked with <<<1,256>>>
3. speed up from 872.88ms to 207.25ms for `convolution_layer_cuda(float*, float*, float*)`
4. now the new bottleneck is the CUDA memcpy part (host to device)
```
==28192== NVPROF is profiling process 28192, command: conv.exe
==28192== Profiling application: conv.exe
==28192== Warning: 7 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==28192== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   96.19%  5.30244s         2  2.65122s  12.672us  5.30243s  [CUDA memcpy HtoD]
                    3.76%  207.08ms         1  207.08ms  207.08ms  207.08ms  convolution_layer_cuda(float*, float*, float*)
                    0.05%  2.7319ms         1  2.7319ms  2.7319ms  2.7319ms  [CUDA memcpy DtoH]
      API calls:   50.48%  207.25ms         1  207.25ms  207.25ms  207.25ms  cudaDeviceSynchronize
                   38.68%  158.80ms         4  39.699ms  582.00us  156.98ms  cudaMalloc
                    9.40%  38.608ms         1  38.608ms  38.608ms  38.608ms  cuDevicePrimaryCtxRelease
                    1.21%  4.9680ms         3  1.6560ms  59.100us  2.8595ms  cudaMemcpy
                    0.20%  831.90us         4  207.98us  160.10us  302.40us  cudaFree
                    0.01%  57.600us         1  57.600us  57.600us  57.600us  cudaLaunchKernel
                    0.01%  26.600us         1  26.600us  26.600us  26.600us  cuModuleUnload
                    0.00%  11.900us       101     117ns       0ns     800ns  cuDeviceGetAttribute
                    0.00%  10.400us         1  10.400us  10.400us  10.400us  cuDeviceTotalMem
                    0.00%  3.8000us         3  1.2660us     200ns  3.3000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuDeviceGetName
                    0.00%  1.7000us         2     850ns     100ns  1.6000us  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

```
