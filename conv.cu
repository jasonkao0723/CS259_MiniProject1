#include <iostream>
#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>
#include <time.h>
#include "device_launch_parameters.h"
#include "cuda.h"
#include "dnn.hpp"
using namespace std;




//Define the parameters if not defined externally
#ifndef Sy
#define Sy 1
#define Sx 1
#endif

#ifndef Tnn
//Tiling Sizes
#define Tnn 32
#define Tn 16
#define Ti 32

#define Ty 8
#define Tx 8
#endif

// For VS
#define Nb 1
#define Ny 224
#define Nx 224
#define Kx 3
#define Ky 3
#define Ni 64
#define Nn 64
#define ThreadsPerBlock 8
#define ThreadsPerBlock_z 4
#define PAD 1

const int NYPAD = 2 * PAD + Ny;
const int NXPAD = 2 * PAD + Ny;
const int NYSCL = (Ny + 2 * PAD - Ky) / Sy + 1;
const int NXSCL = (Nx + 2 * PAD - Kx) / Sx + 1;
const int SYNAPSE_SIZE = (1L * Ky * Kx * Nn * Ni);
const int BlockPerGrid = (NXSCL + ThreadsPerBlock - 1) / ThreadsPerBlock;
const int BlockPerGrid_z = (Nb + ThreadsPerBlock_z - 1) / ThreadsPerBlock_z;

VTYPE(*synapse)[Ky][Kx][Nn][Ni];

VTYPE(*neuron_i)[NYPAD][NXPAD][Ni][Nb];
VTYPE(*neuron_n)[Nb][Nn][NYSCL][NXSCL];
VTYPE(*neuron_n2)[Nb][Nn][NYSCL][NXSCL];
VTYPE(*neuron_n3)[Nb][Nn][NYSCL][NXSCL];

void fill_convolution_shared_simple(VTYPE(&synapse)[Ky][Kx][Nn][Ni], VTYPE(&neuron_i)[NYPAD][NXPAD][Ni][Nb]) 
{
	for (int yy = 0; yy < Ky; ++yy)
	{
		for (int xx = 0; xx < Kx; ++xx)
		{
			for (int nn = 0; nn < Nn; ++nn)
			{
				for (int ni = 0; ni < Ni; ++ni)
				{
					synapse[yy][xx][nn][ni] = rand() % 10; //static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
				}
			}
		}
	}
	for (int yy = 0; yy < NYPAD; yy++)
	{
		for (int xx = 0; xx < NXPAD; xx++)
		{
			for (int ni = 0; ni < Ni; ni++)
			{
				for (int nb = 0; nb < Nb; nb++)
				{
					VTYPE fill_val = 0.0f;
					if (yy == 0 || yy == NYPAD - 1 || xx == 0 || xx == NXPAD - 1) {
						fill_val = 0.0f;
					}
					else {
						fill_val = rand() % 10; //static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
					}
					neuron_i[yy][xx][ni][nb] = fill_val;
				}
			}
		}
	}
}

void  convolution_layer(VTYPE(&synapse)[Ky][Kx][Nn][Ni], VTYPE(&neuron_i)[NYPAD][NXPAD][Ni][Nb], VTYPE(&neuron_n)[Nb][Nn][NYSCL][NXSCL])
{
	VTYPE sum[Nn][Nb] = { 0 };

	// — Original code — (excluding nn, ii loops)
	int yout = 0;
	for (int y = 0; y < NYSCL; y += Sy) { // Ny changed to NYSCL
		int xout = 0;
		for (int x = 0; x < NXSCL; x += Sx) { // Nx changed to NXSCL
			for (int n = 0; n < Nn; n++) {
				
				for (int b = 0; b < Nb; b++)
				{
					sum[n][b] = 0;
					// sliding window;
					for (int ky = 0; ky < Ky; ky++) 
					{
						for (int kx = 0; kx < Kx; kx++)
						{
							for (int i = 0; i < Ni; i++)
							{
								VTYPE sv = synapse[ky][kx][n][i];
								VTYPE nv = neuron_i[ky + y][kx + x][i][b];
								sum[n][b] += sv * nv;
							}
						}
					}
					neuron_n[b][n][yout][xout] = transfer(sum[n][b]);
				}
			}
			xout++;
		}
		yout++;
	}
}

__global__ void convolution_layer_cuda_2DwithTiling(VTYPE* synapse, VTYPE* neuron_i, VTYPE* neuron_n)
{
	VTYPE sum[Nn][Nb] = { 0 };
	// — Original code — (excluding nn, ii loops)
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx < NXSCL && ty < NYSCL) {
#pragma unroll
		for (int nn = 0; nn < Nn; nn += Tn)
		{
#pragma unroll
			for (int n = nn; n < nn + Tn; n++)
			{
				for (int b = 0; b < Nb; b++)
				{
					sum[n][b] = 0;
				}
			}

			// sliding window;
#pragma unroll
			for (int ky = 0; ky < Ky; ky++)
#pragma unroll
				for (int kx = 0; kx < Kx; kx++)
#pragma unroll
					for (int n = nn; n < nn + Tn; n++)
#pragma unroll
						for (int i = 0; i < Ni; i++)
						{
							for (int b = 0; b < Nb; b++)
							{
								VTYPE sv = synapse[ky * (Kx * Nn * Ni) + kx * (Nn * Ni) + n * Ni + i]; //[ky][kx][n][i];
								VTYPE nv = neuron_i[(ky + ty) * (NXPAD * Ni * Nb) + (kx + tx) * (Ni * Nb) + i * Nb + b]; //[ky + y][kx + x][i];
								sum[n][b] += sv * nv;
							}
						}
#pragma unroll
			for (int n = nn; n < nn + Tn; n++)
			{
				for (int b = 0; b < Nb; b++)
				{
					neuron_n[b * (NXSCL * NYSCL * Nn) + n * (NXSCL * NYSCL) + ty * NXSCL + tx] = sum[n][b];
				}
			}
		}
	}
	
}


int main(const int argc, const char** argv)
{
	cout << "Allocating memory\n";

	synapse = (VTYPE(*)[Ky][Kx][Nn][Ni])malloc(SYNAPSE_SIZE * sizeof(VTYPE));
	neuron_i = (VTYPE(*)[NYPAD][NXPAD][Ni][Nb])malloc(NYPAD * NXPAD * Ni * Nb * sizeof(VTYPE));
	neuron_n = (VTYPE(*)[Nb][Nn][NYSCL][NXSCL])malloc(NYSCL * NXSCL * Nn * Nb * sizeof(VTYPE));
	neuron_n2 = (VTYPE(*)[Nb][Nn][NYSCL][NXSCL])malloc(NYSCL * NXSCL * Nn * Nb * sizeof(VTYPE));
	neuron_n3 = (VTYPE(*)[Nb][Nn][NYSCL][NXSCL])malloc(NYSCL * NXSCL * Nn * Nb * sizeof(VTYPE));

	cout << "Allocating memory for CUDA \n";
	VTYPE* d_synapse = NULL;
	VTYPE* d_neuron_i = NULL;
	VTYPE* d_neuron_n = NULL;

	cudaMalloc((void**)&d_synapse, SYNAPSE_SIZE * sizeof(VTYPE));
	cudaMalloc((void**)&d_neuron_i, NYPAD * NXPAD * Ni * Nb * sizeof(VTYPE));
	cudaMalloc((void**)&d_neuron_n, NYSCL * NXSCL * Nn * Nb * sizeof(VTYPE));

	cout << "Initializing arrays\n";

	fill_convolution_shared_simple(*synapse, *neuron_i);
	
	////DEBUG///////////////////////////////////////////////////////////////////////////////
	/*
	for (int n = 0; n < Nn; n++) {
		printf("Output channel, n=%d\n", n);
		for (int i = 0; i < Ni; i++) {
			printf("Input channel, i=%d\n", i);
			for (int y = 0; y < Ky; y++) {
				cout << "|";
				for (int x = 0; x < Kx; x++) {
					cout << (*synapse)[y][x][n][i] << "|";
				}
				cout << endl;
			}
			cout << endl;
		}
	}
	for (int b = 0; b < Nb; b++)
	{
		printf("Input number b=%d\n", b);
		for (int i = 0; i < Ni; i++) 
		{
			printf("Input channel, i=%d\n", i);
			for (int y = 0; y < NYPAD; y++) 
			{
				cout << "|";
				for (int x = 0; x < NXPAD; x++) 
				{
					cout << (*neuron_i)[y][x][i][b] << "|";
				}
				cout << endl;
			}
			cout << endl;
		}
	}
	//for (int i = 0; i < Nn; i++) {
	//	for (int y = 0; y < NYSCL; y++) {
	//		for (int x = 0; x < NXSCL; x++) {
	//			if (x == 4 || y == 4 || i == 4) {
	//				(*neuron_n)[y][x][i] = 69;
	//			}
	//		}
	//	}
	//}
	*/
	////////////////////////////////////////////////////////////////////////////////////////
	

	cout << "Copying initialized arrays from host to device\n";

	cudaMemcpy(d_synapse, synapse, SYNAPSE_SIZE * sizeof(VTYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(d_neuron_i, neuron_i, NYPAD * NXPAD * Ni * Nb * sizeof(VTYPE), cudaMemcpyHostToDevice);
	cout << "Copy from Host to Device: " << cudaGetErrorString(cudaGetLastError()) << endl;

	cout << "Starting computation\n";

	// simple Version
	convolution_layer(*synapse, *neuron_i, *neuron_n);
	cout << "Simple version complete!\n";


	// 2D threads CUDA version with tiling
	convolution_layer_cuda_2DwithTiling <<<dim3(BlockPerGrid, BlockPerGrid), dim3(ThreadsPerBlock, ThreadsPerBlock) >> > (d_synapse, d_neuron_i, d_neuron_n);
	cudaMemcpy(neuron_n2, d_neuron_n, NYSCL * NXSCL * Nn * Nb * sizeof(VTYPE), cudaMemcpyDeviceToHost);
	cout << "2DwithTiling: " << cudaGetErrorString(cudaGetLastError()) << endl;


	cout << "Cuda versions complete!\n";

	// verify the results
	compare((VTYPE*)*neuron_n, (VTYPE*)*neuron_n2, NYSCL, NXSCL, Nn, Nb);
	//compare((VTYPE*)*neuron_n, (VTYPE*)*neuron_n3, NYSCL, NXSCL, Nn);


	free(synapse);
	free(neuron_i);
	free(neuron_n);
	free(neuron_n2);
	free(neuron_n3);
	cudaFree(d_synapse);
	cudaFree(d_neuron_i);
	cudaFree(d_neuron_n);

	cout << "done\n";
}