#include <iostream>
#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>
#include <time.h>
#include "device_launch_parameters.h"
#include "cuda.h"
#include "dnn.hpp"
using namespace std;

#define testsize 64

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
#define Ny 224
#define Nx 224
#define Kx 3
#define Ky 3
#define Ni testsize
#define Nn testsize
#define ThreadsPerBlock 16
#define ThreadsPerBlock_z 8
#define PAD 1

const int NYPAD = 2 * PAD + Ny;
const int NXPAD = 2 * PAD + Ny;
const int NYSCL = (Ny + 2 * PAD - Ky) / Sy + 1;
const int NXSCL = (Nx + 2 * PAD - Kx) / Sx + 1;
const int SYNAPSE_SIZE = (1L * Ky * Kx * Nn * Ni);
const int BlockPerGrid = (NXSCL + ThreadsPerBlock - 1) / ThreadsPerBlock;
const int BlockPerGrid_z = (Nn + ThreadsPerBlock_z - 1) / ThreadsPerBlock_z;

VTYPE(*synapse)[Nn][Ni][Ky][Kx];

VTYPE(*neuron_i)[NYPAD][NXPAD][Ni];
VTYPE(*neuron_n)[Nn][NYSCL][NXSCL];
VTYPE(*neuron_n2)[Nn][NYSCL][NXSCL];
VTYPE(*neuron_n3)[Nn][NYSCL][NXSCL];

inline void errChecking(cudaError_t res) {
	if (res != cudaSuccess) {
		printf("Error: %s", cudaGetErrorString(res));
	}
}

void fill_convolution_shared_simple(VTYPE(&synapse)[Nn][Ni][Ky][Kx], VTYPE(&neuron_i)[NYPAD][NXPAD][Ni]){
	int cnt = 0;
	for (int nn = 0; nn < Nn; ++nn)
	{
		for (int ni = 0; ni < Ni; ++ni)
		{
			for (int yy = 0; yy < Ky; ++yy)
			{
				for (int xx = 0; xx < Kx; ++xx)
				{
					synapse[nn][ni][yy][xx] = cnt;// static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
					cnt++;
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
				VTYPE fill_val = 0.0f;
				if (yy == 0 || yy == NYPAD - 1 || xx == 0 || xx == NXPAD - 1) {
					fill_val = 0.0f;
				}
				else {
					fill_val = 1;// static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
				}
				neuron_i[yy][xx][ni] = fill_val;
			}
		}
	}
}

void  convolution_layer(VTYPE(&synapse)[Nn][Ni][Ky][Kx], VTYPE(&neuron_i)[NYPAD][NXPAD][Ni], VTYPE(&neuron_n)[Nn][NYSCL][NXSCL]) {
	VTYPE sum[Nn] = { 0 };

	// — Original code — (excluding nn, ii loops)
	int yout = 0;
	for (int y = 0; y < NYSCL; y += Sy) { // Ny changed to NYSCL
		int xout = 0;
		for (int x = 0; x < NXSCL; x += Sx) { // Nx changed to NXSCL
			for (int n = 0; n < Nn; n++) {
				sum[n] = 0;

				// sliding window;
				for (int ky = 0; ky < Ky; ky++)
					for (int kx = 0; kx < Kx; kx++)
						for (int i = 0; i < Ni; i++) {
							VTYPE sv = synapse[n][i][ky][kx];
							VTYPE nv = neuron_i[ky + y][kx + x][i];
							sum[n] += sv * nv;
						}
				neuron_n[n][yout][xout] = sum[n];
			}
			xout++;
		}
		yout++;
	}
}

__global__ void convolution_layer_cuda_2DwithTiling(VTYPE* synapse, VTYPE* neuron_i, VTYPE* neuron_n)
{
	VTYPE sum[Nn] = { 0 };
	// — Original code — (excluding nn, ii loops)
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx < NXSCL && ty < NXSCL) {
		for (int nn = 0; nn < Nn; nn += Tn)
		{
			for (int n = nn; n < nn + Tn; n++)
			{
				sum[n] = 0;

			}

			// sliding window;
			for (int ky = 0; ky < Ky; ky++)
				for (int kx = 0; kx < Kx; kx++)
					for (int n = nn; n < nn + Tn; n++)
						for (int i = 0; i < Ni; i++)
						{
							VTYPE sv = synapse[ky * (Kx * Nn * Ni) + kx * (Nn * Ni) + n * Ni + i]; //[ky][kx][n][i];
							VTYPE nv = neuron_i[(ky + ty) * (NXPAD * Ni) + (kx + tx) * Ni + i]; //[ky + y][kx + x][i];
							sum[n] += sv * nv;
						}
			for (int n = nn; n < nn + Tn; n++)
			{
				neuron_n[ty * (NXSCL * Nn) + tx * Nn + n] = (sum[n] > 0) ? sum[n] : sum[n] / 4;
				//printf("yout: %d, xout: %d, index: %d, neuron:%f\n", y, x, index, neuron_n[y * (NXSCL * Nn) + x * Nn + n]);
			}
		}
	}
}

__global__ void convolution_layer_cuda_2DwithTilingSH(VTYPE* synapse, VTYPE* neuron_i, VTYPE* neuron_n)
{

	VTYPE sum[Nn] = { 0 };
	// — Original code — (excluding nn, ii loops)

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int offset1 = Kx; 
	int offset2 = Kx*Ky; 
	int offset3 = Kx*Ky*Ni;

	__shared__ VTYPE kernel[Tn][Ti][Ky][Kx];
	//__shared__ VTYPE neuron[NY][NX][Ti];

	for (int nn = 0; nn < Nn; nn += Tn)
	{
		for (int n = nn; n < nn + Tn; n++)
		{
			sum[n] = 0;
		}

		for (int ii = 0; ii < Ni; ii += Ti)
		{
			// Tiling for kernel and store in SPAD
			for (int n = nn; n < nn + Tn; n++)
			{
				for (int i = ii; i < ii + Ti; i++)
				{
					if (tx < Kx && ty < Ky)
					{
						kernel[n - nn][i - ii][ty][tx] = synapse[offset3 * n + offset2 * i + offset1 * ty + tx];
					}
				}
			}

			__syncthreads();

			if (col < NXSCL && row < NYSCL)
			{
				// sliding window;
				for (int n = nn; n < nn + Tn; n++)
				{
					for (int i = ii; i < ii + Ti; i++)
					{
						for (int ky = 0; ky < Ky; ky++)
						{
							for (int kx = 0; kx < Kx; kx++)
							{
								VTYPE sv = kernel[n - nn][i - ii][ky][kx];						//synapse[ky * (Kx * Nn * Ni) + kx * (Nn * Ni) + n * Ni + i]; //
								VTYPE nv = neuron_i[(ky + row) * (NXPAD * Ni) + (kx + col) * Ni + i];				//neuron_i[(ky + row) * (NXPAD * Ni) + (kx + col) * Ni + i]; // neuron[ky][kx][i - ii]; // 
								sum[n] += sv * nv;
								//if (row == 0 && col == 0 && n == 0 && kx==1 && ky==1)
								//{
								//	printf("channel(i): %d \t (ky:%d, kx:%d)\t sv*nv=%lf\n", i, ky, kx, sv * nv);
								//	printf("cum sum:%lf\n", sum[n]);
								//}
							}

						}
					}
				}
				for (int n = nn; n < nn + Tn; n++)
				{
					neuron_n[n * (NXSCL * NYSCL) + row * NXSCL + col] = sum[n];
				}
			}
			__syncthreads();
		}
	}
}


int main(const int argc, const char** argv)
{
	cout << "Allocating memory\n";

	synapse = (VTYPE(*)[Nn][Ni][Ky][Kx])malloc(SYNAPSE_SIZE * sizeof(VTYPE));
	neuron_i = (VTYPE(*)[NYPAD][NXPAD][Ni])malloc(NYPAD * NXPAD * Ni * sizeof(VTYPE));
	neuron_n = (VTYPE(*)[Nn][NYSCL][NXSCL])malloc(NYSCL * NXSCL * Nn * sizeof(VTYPE));
	neuron_n2 = (VTYPE(*)[Nn][NYSCL][NXSCL])malloc(NYSCL * NXSCL * Nn * sizeof(VTYPE));
	neuron_n3 = (VTYPE(*)[Nn][NYSCL][NXSCL])malloc(NYSCL * NXSCL * Nn * sizeof(VTYPE));

	cout << "Allocating memory for CUDA \n";
	VTYPE* d_synapse = NULL;
	VTYPE* d_neuron_i = NULL;
	VTYPE* d_neuron_n = NULL;

	cudaMalloc((void**)&d_synapse, SYNAPSE_SIZE * sizeof(VTYPE));
	cudaMalloc((void**)&d_neuron_i, NYPAD * NXPAD * Ni * sizeof(VTYPE));
	cudaMalloc((void**)&d_neuron_n, NYSCL * NXSCL * Nn * sizeof(VTYPE));

	cout << "Initializing arrays\n";

	fill_convolution_shared_simple(*synapse, *neuron_i);
	
	////DEBUG///////////////////////////////////////////////////////////////////////////////
	/*
	for (int n = 0; n < 1; n++) {
		printf("Output channel %d\n", n);
		for (int i = 0; i < Ni; i++) {
			printf("Input channel %d\n", i);
			for (int y = 0; y < Ky; y++) {
				for (int x = 0; x < Kx; x++) {
					cout << (*synapse)[n][i][y][x];
				}
				cout << endl;
			}
			cout << endl;
		}
	}
	
	for (int i = 0; i < Ni; i++) {
		printf("Input channel %d\n", i);
		for (int y = 0; y < NYPAD; y++) {
			for (int x = 0; x < NXPAD; x++) {
				cout << (*neuron_i)[y][x][i];
			}
			cout << endl;
		}
		cout << endl;
	}
	for (int i = 0; i < Nn; i++) {
		for (int y = 0; y < NYSCL; y++) {
			for (int x = 0; x < NXSCL; x++) {
				if (x == 4 || y == 4 || i == 4) {
					(*neuron_n)[y][x][i] = 69;
				}
			}
		}
	}
	////////////////////////////////////////////////////////////////////////////////////////
	*/

	cout << "Copying initialized arrays from host to device\n";

	cudaMemcpy(d_synapse, synapse, SYNAPSE_SIZE * sizeof(VTYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(d_neuron_i, neuron_i, NYPAD * NXPAD * Ni * sizeof(VTYPE), cudaMemcpyHostToDevice);
	cout << "Copy from Host to Device: " << cudaGetErrorString(cudaGetLastError()) << endl;


	cout << "Starting computation\n";
	// simple Version
	convolution_layer(*synapse, *neuron_i, *neuron_n);
	cout << "Simple version complete!\n";

	// 2D threads CUDA version with tiling
	//convolution_layer_cuda_2DwithTiling<<<dim3(BlockPerGrid, BlockPerGrid), dim3(ThreadsPerBlock, ThreadsPerBlock)>>>(d_synapse, d_neuron_i, d_neuron_n);
	//cudaMemcpy(neuron_n2, d_neuron_n, NYSCL * NXSCL * Nn * sizeof(VTYPE), cudaMemcpyDeviceToHost);
	//cout << "2DwithTiling: " << cudaGetErrorString(cudaGetLastError()) << endl;


		//// 2D threads CUDA version with tiling and shared memory
	convolution_layer_cuda_2DwithTilingSH <<<dim3(BlockPerGrid, BlockPerGrid), dim3(ThreadsPerBlock, ThreadsPerBlock)>>> (d_synapse, d_neuron_i, d_neuron_n);
	cudaMemcpy(neuron_n3, d_neuron_n, NYSCL * NXSCL * Nn * sizeof(VTYPE), cudaMemcpyDeviceToHost);
	cout << "2DwithTilingSH: " << cudaGetErrorString(cudaGetLastError()) << endl;

	// verify the results
	//compare((VTYPE*)*neuron_n, (VTYPE*)*neuron_n2, NYSCL, NXSCL, Nn);
	compare((VTYPE*)*neuron_n, (VTYPE*)*neuron_n3, NYSCL, NXSCL, Nn);


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