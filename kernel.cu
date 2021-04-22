
#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <time.h>
#include "device_launch_parameters.h"
#include "cuda.h"
using namespace std;

#define Nn 4096
#define Ni 25088
#define BATCH_SIZE 1
#define BLOCK_SIZE 32
#define VTYPE double

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
    
    for (int i = 0; i < Ni; i++) {
        h_neuron_i[i] = rand() / (VTYPE)RAND_MAX;
    }

    for (int i = 0; i < Ni * Nn; i++) {
        synapse[i] = rand() / (VTYPE)RAND_MAX;
    }
}

void h_MatMul(VTYPE* h_neuron_i, VTYPE* h_neuron_n, VTYPE* synapse) {
    for (int i = 0; i < Nn; i++) {
        VTYPE temp = 0.0f; 
        for (int j = 0; j < Ni; j++) {
            temp += h_neuron_i[j] * synapse[i*Nn+j];
        }
        h_neuron_n[i] = temp;
    }
}

__global__ void d_MatMul_simple1(const VTYPE* d_neuron_i, VTYPE* d_neuron_n, const VTYPE* synapse) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    
    VTYPE temp = 0.0f;

    #pragma unroll
    for (int i = 0; i < Ni; i++) {
        temp += d_neuron_i[i] * synapse[col * Ni + i];
    }
    d_neuron_n[col] = temp;
}

__global__ void d_MatMul_simple2(const VTYPE* d_neuron_i, VTYPE* d_neuron_n, const VTYPE* d_synapse) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ VTYPE neuron_i[BLOCK_SIZE];

    VTYPE temp = 0.0f;

    for (int i = 0; i < Ni; i += BLOCK_SIZE) {
        // Phase i = 0:195 
        neuron_i[threadIdx.x] = d_neuron_i[i+threadIdx.x];
        // synapse[threadIdx.x] = d_synapse[col + i];

        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; j++) {
            temp += neuron_i[j] * d_synapse[col * Ni + j + i];
        }

        __syncthreads();
    }

    d_neuron_n[col] = temp;    
}


__global__ void d_MatMul_simple3(const VTYPE* d_neuron_i, VTYPE* d_neuron_n, const VTYPE* d_synapse) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ VTYPE synapse[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ VTYPE neuron_i[BLOCK_SIZE];

    VTYPE temp = 0.0f;

    for (int i = 0; i < Ni; i += BLOCK_SIZE) {
        // Phase i = 0:195 
        neuron_i[threadIdx.x] = d_neuron_i[i + threadIdx.x];
        // synapse[threadIdx.x] = d_synapse[col + i];

        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; j++) {
            temp += neuron_i[j] * d_synapse[col * Ni + j + i];
        }

        __syncthreads();

    }

    d_neuron_n[col] = temp;
}

__global__ void d_test(VTYPE* d_synapse, VTYPE* d_neuron_i) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    d_neuron_i[idx] *= 1.1f;
}


bool compare(VTYPE* neuron1, VTYPE* neuron2) {
    bool good = true;

    for (int i = 0; i < Nn; i++) {
        if (fabs(neuron1[i] - neuron2[i]) > 1e-2)
        {
            good = false;
            printf("At index %d \t Host result: %lf \t Device result: %lf \n", i, neuron1[i], neuron2[i]);
        }

        
    }

    return good;
}

int main()
{
    // Initialize arrays on host
    VTYPE* h_neuron_i = (VTYPE*)malloc(Ni * sizeof(VTYPE));
    VTYPE* h_neuron_n = (VTYPE*)malloc(Nn * sizeof(VTYPE));
    VTYPE* h_synapse = (VTYPE*)malloc(Nn * Ni * sizeof(VTYPE));
    VTYPE* h_neuron_n2 = (VTYPE*)malloc(Nn * sizeof(VTYPE));

    init_layer(h_neuron_i, h_neuron_n, h_synapse);
    
    // Compute Matrix Vector Multiplication on host
    h_MatMul(h_neuron_i, h_neuron_n, h_synapse);

    // Allocate memory on device
    VTYPE* d_neuron_i = NULL;
    VTYPE* d_neuron_n = NULL;
    VTYPE* d_synapse = NULL;
    
    cudaMalloc((void**)&d_neuron_i, Ni * sizeof(VTYPE));
    cudaMalloc((void**)&d_neuron_n, Nn * sizeof(VTYPE));
    cudaMalloc((void**)&d_synapse, Nn * Ni * sizeof(VTYPE));

    // Copy arrays from host to device
    cudaMemcpy(d_neuron_i, h_neuron_i, Ni * sizeof(VTYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neuron_n, h_neuron_n, Nn * sizeof(VTYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_synapse, h_synapse, Nn * Ni * sizeof(VTYPE), cudaMemcpyHostToDevice);


     //Define kernel launch parameters
    int ThreadsPerBlock = BLOCK_SIZE;
    int BlocksPerGrid = (Nn + ThreadsPerBlock - 1) / ThreadsPerBlock;
     //Launch kernel
    d_MatMul_simple1<<<BlocksPerGrid, ThreadsPerBlock>>>(d_neuron_i, d_neuron_n, d_synapse);

    //Define kernel launch parameters
    ThreadsPerBlock = BLOCK_SIZE;
    BlocksPerGrid = (Nn + ThreadsPerBlock - 1) / ThreadsPerBlock;
    //Launch kernel
    d_MatMul_simple2<<<BlocksPerGrid, ThreadsPerBlock>>>(d_neuron_i, d_neuron_n, d_synapse);


    //Define kernel launch parameters
    ThreadsPerBlock = BLOCK_SIZE;
    BlocksPerGrid = (Nn + ThreadsPerBlock - 1) / ThreadsPerBlock;
    //Launch kernel
    d_MatMul_simple3<<<BlocksPerGrid, ThreadsPerBlock >>>(d_neuron_i, d_neuron_n, d_synapse);



    // Run and time on host    
    clock_t begin = clock();
    for (int i = 0; i < Nn; i++) {
        VTYPE temp = 0.0f;
        for (int j = 0; j < Ni; j++) {
            temp += h_neuron_i[j] * h_synapse[i * Ni + j];
        }
        h_neuron_n[i] = temp;
    }
    double elapsed = ((double)clock() - (double)begin) / (double)CLOCKS_PER_SEC;
    printf("Took CPU %lf seconds to run\n", elapsed);

    // Copy results from device back to host
    cudaMemcpy(h_neuron_n2, d_neuron_n, Nn * sizeof(VTYPE), cudaMemcpyDeviceToHost);


    /*
    VTYPE temp = 0.0f;
    int k = 0;
    int phase = 1;
    for (int i = phase * BLOCK_SIZE; i < (phase + 1) * BLOCK_SIZE; i++) {
        temp += h_synapse[k * Ni + i];
    }
    printf("temp in host : %lf\n", temp);
    //h_neuron_i[i] * 
    
    */

    // Compare host and device results
    if (compare(h_neuron_n, h_neuron_n2)) {
        printf("Passed!\n");
    }

    // Free up memory
    cudaFree(d_neuron_i);
    cudaFree(d_neuron_n);
    cudaFree(d_synapse);
    free(h_neuron_i);
    free(h_neuron_n);
    free(h_synapse);
    free(h_neuron_n2);

    return 0;
}

