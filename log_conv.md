###
1. 3DwithTiling: still a WIP.
2. Best result so far: 3.8038ms for 2D threads with tiling
3. Batching to be added. 

'''
==179153== NVPROF is profiling process 179153, command: ./a.out
Initializing arrays
Copying initialized arrays from host to device
Copy from Host to Device: no error
Starting computation
Simple version complete!
2DwithoutTiling: no error
1DwithoutTiling: no error
2DwithTiling: no error
1DwithTiling: no error
3DwithTiling: invalid argument
Cuda versions complete!
RESULTS MATCH!
RESULTS MATCH!
RESULTS MATCH!
RESULTS MATCH!
done
==179153== Profiling application: ./a.out
==179153== Warning: 1 records have invalid timestamps due to insufficient device buffer space. You can configure the buffer space using the option --device-buffer-size.
==179153== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   54.82%  279.16ms         1  279.16ms  279.16ms  279.16ms  convolution_layer_cuda_1DwithoutTiling(float*, float*, float*)
                   34.46%  175.49ms         1  175.49ms  175.49ms  175.49ms  convolution_layer_cuda_1DwithTiling(float*, float*, float*)
                    4.83%  24.616ms         1  24.616ms  24.616ms  24.616ms  convolution_layer_cuda_2DwithoutTiling(float*, float*, float*)
                    4.63%  23.580ms         4  5.8950ms  5.6669ms  6.1610ms  [CUDA memcpy DtoH]
                    0.75%  3.8038ms         1  3.8038ms  3.8038ms  3.8038ms  convolution_layer_cuda_2DwithTiling(float*, float*, float*)
                    0.51%  2.5905ms         2  1.2952ms  15.359us  2.5751ms  [CUDA memcpy HtoD]
      API calls:   51.68%  554.64ms         3  184.88ms  124.81us  554.38ms  cudaMalloc
                   47.87%  513.78ms         7  73.397ms  1.3620us  286.11ms  cudaMemcpy
                    0.18%  1.9700ms        96  20.520us     132ns  830.38us  cuDeviceGetAttribute
                    0.12%  1.2543ms         3  418.10us  163.86us  912.90us  cudaFree
                    0.07%  791.23us         1  791.23us  791.23us  791.23us  cuDeviceTotalMem
                    0.06%  626.65us         5  125.33us  30.538us  364.22us  cudaLaunchKernel
                    0.01%  108.32us         1  108.32us  108.32us  108.32us  cuDeviceGetName
                    0.00%  6.4730us         1  6.4730us  6.4730us  6.4730us  cuDeviceGetPCIBusId
                    0.00%  3.5170us         6     586ns     125ns  1.5210us  cudaGetLastError
                    0.00%  2.7550us         2  1.3770us     211ns  2.5440us  cuDeviceGet
                    0.00%  1.7600us         6     293ns     167ns     657ns  cudaGetErrorString
                    0.00%  1.6440us         3     548ns     170ns  1.1650us  cuDeviceGetCount
'''
