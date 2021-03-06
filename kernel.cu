#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>
#include <time.h>
#include "device_launch_parameters.h"
#include "cuda.h"
using namespace std;

#define Nn 4096
#define Ni 25088
#define BATCH_SIZE 16
#define BLOCK_SIZE 32
#define BlockSize2D 16
#define VTYPE float

/*  
*   synapse (w) is (Nn x Ni)^T 
*   neuron_i (x) is (BATCH_SIZE x Ni)
*   neuron_n (y) is (BATCH_SIZE x Nn)
*   
*   y = Xw^T
*/

void init_layer(VTYPE* h_neuron_i, VTYPE* h_neuron_n, VTYPE* synapse) {
    for (int i = 0; i < Nn; i++) {
        h_neuron_n[i] = rand() / (VTYPE)RAND_MAX;
    }
    
    for (int i = 0; i < Ni * BATCH_SIZE; i++) {
        h_neuron_i[i] = rand() / (VTYPE)RAND_MAX;
    }

    for (int i = 0; i < Ni * Nn; i++) {
        synapse[i] = rand() / (VTYPE)RAND_MAX;
    }
}

__launch_bounds__(1024,2)
__global__ void d_MatMul_simple1(const VTYPE* d_neuron_i, VTYPE* d_neuron_n, const VTYPE* synapse) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    VTYPE temp = 0.0f;
    
    if (col < Nn && row < BATCH_SIZE) {
        #pragma unroll
        for (int i = 0; i < Ni; i++) {
            temp += d_neuron_i[row * Ni + i] * synapse[col + Nn * i];
        }
        d_neuron_n[row * Nn + col] = temp;
    }
}

__global__ void d_MatMul_simple2(const VTYPE* d_neuron_i, VTYPE* d_neuron_n, const VTYPE* d_synapse) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ VTYPE neuron_i[BlockSize2D][BlockSize2D];

    VTYPE temp = 0.0f;

    #pragma unroll
    for (int i = 0; i < Ni; i += BlockSize2D) {

        if (row < BATCH_SIZE && i + threadIdx.x < Ni) {
            neuron_i[threadIdx.y][threadIdx.x] = d_neuron_i[row * Ni + i + threadIdx.x];
        }
        else {
            neuron_i[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < BlockSize2D; j++) {
            temp += neuron_i[threadIdx.y][j] * d_synapse[(j + i)* Nn + col];
        }
    
        __syncthreads();
    }
    if (col < Nn && row < BATCH_SIZE) {
        d_neuron_n[row * Nn + col] = temp;
    }
    
}

__global__ void d_MatMul_simple3(const VTYPE* d_neuron_i, VTYPE* d_neuron_n, const VTYPE* d_synapse) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ VTYPE synapse[BlockSize2D][BlockSize2D];
    __shared__ VTYPE neuron[BlockSize2D][BlockSize2D];

    // MxK = MxN * NxK
    VTYPE temp = 0.0f;
    #pragma unroll
    for (int i = 0; i < (Ni - 1) / BlockSize2D + 1; i++) {
        if (row < BATCH_SIZE && i * BlockSize2D + threadIdx.x < Ni) {
            neuron[threadIdx.y][threadIdx.x] = d_neuron_i[row * Ni + i * BlockSize2D + threadIdx.x];
        } 
        else {
            neuron[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (i * BlockSize2D + threadIdx.y < Ni && col < Nn) {
            synapse[threadIdx.y][threadIdx.x] = d_synapse[(i * BlockSize2D + threadIdx.y) * Nn + col];
        }
        else {
            synapse[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();
        #pragma unroll
        for (int j = 0; j < BlockSize2D; j++) {
            temp += neuron[threadIdx.y][j] * synapse[j][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < BATCH_SIZE && col < Nn) {
        d_neuron_n[row * Nn + col] = temp;
    }
}

__global__ void d_test(VTYPE* d_synapse, VTYPE* d_neuron_i) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    d_neuron_i[idx] *= 1.1f;
}

bool compare(VTYPE* neuron1, VTYPE* neuron2) {
    bool good = true;
    #pragma unroll
    for (int k = 0; k < BATCH_SIZE; k++) {
        #pragma unroll
        for (int i = 0; i < Nn; i++) {
            if (fabs(neuron1[k * Nn + i] - neuron2[k * Nn + i]) > 1e-2)
            {
                good = false;
                printf("At index (%d, %d) \t Host result: %lf \t Device result: %lf \n", k, i, neuron1[k * Nn + i], neuron2[k * Nn + i]);
            }
        }
    }
    return good;
}

int main()
{
    // Initialize arrays on host
    VTYPE* h_neuron_i = (VTYPE*)malloc(Ni * BATCH_SIZE *sizeof(VTYPE));
    VTYPE* h_neuron_n1 = (VTYPE*)malloc(Nn * BATCH_SIZE *sizeof(VTYPE));
    VTYPE* h_synapse = (VTYPE*)malloc(Nn * Ni * sizeof(VTYPE));
    VTYPE* h_neuron_n2 = (VTYPE*)malloc(Nn * BATCH_SIZE * sizeof(VTYPE));
    VTYPE* h_neuron_n3 = (VTYPE*)malloc(Nn * BATCH_SIZE * sizeof(VTYPE));
    VTYPE* h_neuron_n = (VTYPE*)malloc(Nn * BATCH_SIZE * sizeof(VTYPE));


    init_layer(h_neuron_i, h_neuron_n, h_synapse);
    
    // Allocate memory on device
    VTYPE* d_neuron_i = NULL;
    VTYPE* d_neuron_n1 = NULL;
    VTYPE* d_neuron_n2 = NULL;
    VTYPE* d_neuron_n3 = NULL;
    VTYPE* d_synapse = NULL;
    VTYPE* test_var = NULL;
    
    cudaMalloc((void**)&d_neuron_i, Ni * BATCH_SIZE * sizeof(VTYPE));
    cudaMalloc((void**)&d_neuron_n1, Nn * BATCH_SIZE * sizeof(VTYPE));
    cudaMalloc((void**)&d_neuron_n2, Nn * BATCH_SIZE * sizeof(VTYPE));
    cudaMalloc((void**)&d_neuron_n3, Nn * BATCH_SIZE * sizeof(VTYPE));
    cudaMalloc((void**)&d_synapse, Nn * Ni * sizeof(VTYPE));
    cudaMalloc((void**)&test_var, sizeof(VTYPE));



    // Copy arrays from host to device
    cudaMemcpy(d_neuron_i, h_neuron_i, Ni * BATCH_SIZE * sizeof(VTYPE), cudaMemcpyHostToDevice);
    cout << "Copy from Host to Device: " << cudaGetErrorString(cudaGetLastError()) << endl;
    cudaMemcpy(d_synapse, h_synapse, Nn * Ni * sizeof(VTYPE), cudaMemcpyHostToDevice);
    cout << "Copy from Host to Device: " << cudaGetErrorString(cudaGetLastError()) << endl;

     //Define kernel launch parameters
    dim3 ThreadsPerBlock2D = dim3(BlockSize2D, BlockSize2D);
    dim3 BlocksPerGrid2D = dim3((Nn + BlockSize2D - 1) / BlockSize2D, (BATCH_SIZE + BlockSize2D - 1) / BlockSize2D);
    
    //Launch kernel #1#
    d_MatMul_simple1<<<BlocksPerGrid2D, ThreadsPerBlock2D>>>(d_neuron_i, d_neuron_n1, d_synapse);
    cout << "MatMul_simple1: " << cudaGetErrorString(cudaGetLastError()) << endl;

    // Copy results from device back to host
    cudaMemcpy(h_neuron_n1, d_neuron_n1, Nn * BATCH_SIZE * sizeof(VTYPE), cudaMemcpyDeviceToHost);
    cout << "Copy from Device to Host: " << cudaGetErrorString(cudaGetLastError()) << endl;

    //Launch kernel #2#
    d_MatMul_simple2<<<BlocksPerGrid2D, ThreadsPerBlock2D >>>(d_neuron_i, d_neuron_n2, d_synapse);
    cout << "MatMul_simple1: " << cudaGetErrorString(cudaGetLastError()) << endl;

    // Copy results from device back to host
    cudaMemcpy(h_neuron_n2, d_neuron_n2, Nn * BATCH_SIZE * sizeof(VTYPE), cudaMemcpyDeviceToHost);
    cout << "Copy from Device to Host: " << cudaGetErrorString(cudaGetLastError()) << endl;

    //Launch kernel #3#
    d_MatMul_simple3<<<BlocksPerGrid2D, ThreadsPerBlock2D >>>(d_neuron_i, d_neuron_n3, d_synapse);
    cout << "MatMul_simple1: " << cudaGetErrorString(cudaGetLastError()) << endl;

    // Copy results from device back to host
    cudaMemcpy(h_neuron_n3, d_neuron_n3, Nn * BATCH_SIZE * sizeof(VTYPE), cudaMemcpyDeviceToHost);
    cout << "Copy from Device to Host: " << cudaGetErrorString(cudaGetLastError()) << endl;

    // Run and time on host    
    clock_t begin = clock();
    #pragma unroll
    for (int k = 0; k < BATCH_SIZE; k++) {
        #pragma unroll
        for (int i = 0; i < Nn; i++) {
            VTYPE temp = 0.0f;
            #pragma unroll
            for (int j = 0; j < Ni; j++) {
                temp += h_neuron_i[k * Ni + j] * h_synapse[i + Nn * j];
            }
            h_neuron_n[k * Nn + i] = temp;
        }
        /* 
        *  h_neuron_i  16 x 25088
        *  h_synapse 4096 x 25088
        *  h_neuron_n 16 x 4096
        */
    }

    double elapsed = ((double)clock() - (double)begin) / (double)CLOCKS_PER_SEC;
    printf("Took CPU %lf seconds to run\n", elapsed);

    /*
    VTYPE temp = 0.0f;
    int k = 1;
    int phase = 0;
    for (int i = phase * BlockSize2D; i < (phase + 1) * BlockSize2D; i++) {
        temp += h_synapse[k * Ni + i];
    }
    printf("temp in host : %lf\n", temp);
    */

     //Compare host and device results
    if (compare(h_neuron_n, h_neuron_n1)) {
        printf("1 Passed!\n");
    }
    if (compare(h_neuron_n, h_neuron_n2)) {
        printf("2 Passed!\n");
    }
    if (compare(h_neuron_n, h_neuron_n3)) {
        printf("3 Passed!\n");
    }

    cout << "Host output[0][6:9]: ";
    for (int i = 6; i < 9; i++) {
        printf("%lf, ", h_neuron_n[i]);
    }
    cout << endl;
    cout << "Kernel1 output[0][6:9]: ";
    for (int i = 6; i < 9; i++) {
        printf("%lf, ", h_neuron_n1[i]);
    }
    cout << endl;
    cout << "Kernel2 output[0][6:9]: ";
    for (int i = 6; i < 9; i++) {
        printf("%lf, ", h_neuron_n2[i]);
    }
    cout << endl;
    cout << "Kernel3 output[0][6:9]: ";
    for (int i = 6; i < 9; i++) {
        printf("%lf, ", h_neuron_n3[i]);
    }
    cout << endl;

    // Free up memory
    cudaFree(d_neuron_i);
    cudaFree(d_neuron_n1);
    cudaFree(d_neuron_n2);
    cudaFree(d_neuron_n3);
    cudaFree(d_synapse);
    cudaFree(test_var);
    free(h_neuron_i);
    free(h_neuron_n);
    free(h_synapse);
    free(h_neuron_n1);
    free(h_neuron_n2);
    free(h_neuron_n3);

    cout << "done\n";

    return 0;
}

