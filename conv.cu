#include <iostream>
#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>
#include <time.h>
#include "device_launch_parameters.h"
#include "cuda.h"
#include "dnn.hpp"
using namespace std;

/*
// Input image size
#define Nx 5
#define Ny 5
#define Ni 3

// Filter size
#define Kx 2
#define Ky 2
#define Nn 2

// Other params
#define STRIDE 1
#define PAD 1

#define BATCH_SIZE 1
#define BLOCK_SIZE 32
#define VTYPE float

const int NYPAD = 2 * PAD + Ny;
const int NXPAD = 2 * PAD + Nx;
const int Ox = (Nx + 2 * PAD - Kx) / STRIDE + 1;
const int Oy = (Ny + 2 * PAD - Ky) / STRIDE + 1;

// Output_x = (Nx + 2 * PAD - Kx) / STRIDE + 1
// Output_y = (Ny + 2 * PAD - Ky) / STRIDE + 1
// Output_z = Nn 


void init_layer(VTYPE* h_neuron_i, VTYPE* h_neuron_n, VTYPE* h_synapse) {
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int i = 0; i < Ni; i++) {
            for (int y = 0; y < NYPAD; y++) {
                for (int x = 0; x < NXPAD; x++) {
                    VTYPE fill_val = 0.0f;
                    if (y == 0 || y == NYPAD - 1 || x == 0 || x == NXPAD - 1) {
                        fill_val = 0;
                    }
                    else {
                        fill_val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
                    }
                                     
                    h_neuron_i[b * (NYPAD * NXPAD * Ni) + i * NYPAD * NXPAD + y * NXPAD + x] = fill_val;
                }
            }
        }
    }

    for (int n = 0; n < Nn; n++) {
        for (int i = 0; i < Ni; i++) {
            for (int y = 0; y < Ky; y++) {
                for (int x = 0; x < Kx; x++) {
                    VTYPE fill_val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
                    h_synapse[n * (Kx * Ky * Ni) + i * (Kx * Ky) + y * (Kx)+x] = fill_val;
                }
            }
        }
    }

    for (int i = 0; i < Ox * Oy * Nn * BATCH_SIZE; i++) {
        h_neuron_n[i] = 0;
    }
}


void conv_layer(const VTYPE* h_neuron_i, VTYPE* h_neuron_n, const VTYPE* h_synapse) {
    VTYPE sum[Nn] = { 0 };
    int Tn = 1;
    // — Original code — (excluding nn, ii loops)
    int yout = 0;
    for (int y = 0; y < Ny; y += STRIDE) { // tiling for y;
        int xout = 0;
        for (int x = 0; x < Ny; x += STRIDE) { // tiling for x;
            for (int nn = 0; nn < Nn; nn += Tn) {
                for (int n = nn; n < nn + Tn; n++) {
                    sum[n] = 0;
                }

                // sliding window;
                for (int ky = 0; ky < Ky; ky++)
                    for (int kx = 0; kx < Kx; kx++)
                        for (int n = nn; n < nn + Tn; n++)
                            for (int i = 0; i < Ni; i++) {
                                VTYPE sv = synapse[ky][kx][n][i];
                                VTYPE nv = neuron_i[ky + y][kx + x][i];
                                sum[n] += sv * nv;
                            }
                for (int n = nn; n < nn + Tn; n++) {
                    neuron_n[yout][xout][n] = sum[n];
                }
            }
            xout++;
        }
        yout++;
    }
}


int main() {
    VTYPE* h_neuron_i = (VTYPE*)malloc(NXPAD * NYPAD * Ni * BATCH_SIZE * sizeof(VTYPE));
    VTYPE* h_neuron_n = (VTYPE*)malloc(Ox * Oy * Nn * BATCH_SIZE * sizeof(VTYPE));
    VTYPE* h_synapse = (VTYPE*)malloc(Kx * Ky * Nn * Ni * sizeof(VTYPE));

    // Randomizes filter and input and pads the input
    init_layer(h_neuron_i, h_neuron_n, h_synapse);
    
    // Allocate memory on device
    VTYPE* d_neuron_i = NULL;
    VTYPE* d_neuron_n = NULL;
    VTYPE* d_synapse = NULL;

    cudaMalloc((void**)&d_neuron_i, NXPAD * NYPAD * Ni * BATCH_SIZE * sizeof(VTYPE));
    cudaMalloc((void**)&d_neuron_n, Ox * Oy * Nn * BATCH_SIZE * sizeof(VTYPE));
    cudaMalloc((void**)&d_synapse, Kx * Ky * Nn * Ni * sizeof(VTYPE));

    conv_layer(h_neuron_i, h_neuron_n, h_synapse);

    cout << h_neuron_n[0] << endl;






    // Free up memory
    cudaFree(d_neuron_i);
    cudaFree(d_neuron_n);
    cudaFree(d_synapse);
    free(h_neuron_i);
    free(h_neuron_n);
    free(h_synapse);
    return 0;
}
*/



//Define the parameters if not defined externally
#ifndef Sy
#define Sy 1
#define Sx 1
#endif

#ifndef Tnn
//Tiling Sizes
#define Tnn 32
#define Tn 16
#define Ti 16

#define Ty 8
#define Tx 8
#endif

// For VS
#define Ny 224
#define Nx 224
#define Kx 3
#define Ky 3
#define Ni 64
#define Nn 64
#define ThreadsPerBlock 16
#define ThreadsPerBlock_z 4
#define PAD 1

const int NYPAD = 2 * PAD + Ny;
const int NXPAD = 2 * PAD + Ny;
const int NYSCL = (Ny + 2 * PAD - Ky) / Sy + 1;
const int NXSCL = (Nx + 2 * PAD - Kx) / Sx + 1;
const int SYNAPSE_SIZE = (1L * Ky * Kx * Nn * Ni);
const int BlockPerGrid = (NXSCL + ThreadsPerBlock - 1) / ThreadsPerBlock;
const int BlockPerGrid_z = (Nn + ThreadsPerBlock_z - 1) / ThreadsPerBlock_z;

VTYPE(*synapse)[Ky][Kx][Nn][Ni];

VTYPE(*neuron_i)[NYPAD][NXPAD][Ni];
VTYPE(*neuron_n)[NYSCL][NXSCL][Nn];
VTYPE(*neuron_n2)[NYSCL][NXSCL][Nn];
VTYPE(*neuron_n3)[NYSCL][NXSCL][Nn];
VTYPE(*neuron_n4)[NYSCL][NXSCL][Nn];
VTYPE(*neuron_n5)[NYSCL][NXSCL][Nn];
VTYPE(*neuron_n6)[NYSCL][NXSCL][Nn];

inline void errChecking(cudaError_t res) {
	if (res != cudaSuccess) {
		printf("Error: %s", cudaGetErrorString(res));
	}
}

void fill_convolution_shared_simple(VTYPE(&synapse)[Ky][Kx][Nn][Ni], VTYPE(&neuron_i)[NYPAD][NXPAD][Ni]){
	for (int yy = 0; yy < Ky; ++yy)
	{
		for (int xx = 0; xx < Kx; ++xx)
		{
			for (int nn = 0; nn < Nn; ++nn)
			{
				for (int ni = 0; ni < Ni; ++ni)
				{
					synapse[yy][xx][nn][ni] = 1;// static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
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

void convolution_layer_blocked(	VTYPE(&synapse)[Ky][Kx][Nn][Ni], VTYPE(&neuron_i)[NYPAD][NXPAD][Ni], VTYPE(&neuron_n)[NYSCL][NXSCL][Nn]){
	int c1 = 0, c2 = 0;
	VTYPE sum[Nn] = { 0 };

	for (int yy = 0; yy < Ny; yy += Ty)
	{
		for (int xx = 0; xx < Nx; xx += Tx)
		{
			for (int nnn = 0; nnn < Nn; nnn += Tnn)
			{
				int yout = yy / Sy;
				for (int y = yy; y < yy + Ty; y += Sy)
				{ // tiling for y;
					int xout = xx / Sx;

					for (int x = xx; x < xx + Tx; x += Sx)
					{ // tiling for x;

						for (int nn = nnn; nn < nnn + Tnn; nn += Tn)
						{
							for (int n = nn; n < nn + Tn; n++)
							{
								sum[n] = 0;
							}

							for (int ky = 0; ky < Ky; ky++)
							{ // sliding window;
								for (int kx = 0; kx < Kx; kx++)
								{

									int ii = 0;
									VTYPE sum_sc;

									for (; ii < Ni - Ti + 1; ii += Ti)
									{
										for (int n = nn; n < nn + Tn; n++)
										{
											sum_sc = 0;
											for (int i = ii; i < ii + Ti; i++)
											{
												VTYPE sv = synapse[ky][kx][n][i];
												VTYPE nv = neuron_i[ky + y][kx + x][i];
												sum_sc += sv * nv;
											}
											sum[n] += sum_sc;
										}
									}
								}
							}

							//transfer
							for (int n = nn; n < nn + Tn; n++)
							{
								neuron_n[yout][xout][n] = transfer(sum[n]);
							}
						}
						xout++;
					}
					yout++;
				}
			}
		}
	}
}

void  convolution_layerDN(VTYPE(&synapse)[Ky][Kx][Nn][Ni],
	VTYPE(&neuron_i)[NYPAD][NXPAD][Ni],
	VTYPE(&neuron_n)[NYSCL][NXSCL][Nn]) {
	VTYPE sum[Nn] = { 0 };

	// — Original code — (excluding nn, ii loops)
	int yout = 0;
	for (int y = 0; y < Ny; y += Sy) { // tiling for y;
		int xout = 0;
		for (int x = 0; x < Ny; x += Sx) { // tiling for x;
			for (int nn = 0; nn < Nn; nn += Tn) {
				for (int n = nn; n < nn + Tn; n++) {
					sum[n] = 0;
				}

				// sliding window;
				for (int ky = 0; ky < Ky; ky++)
					for (int kx = 0; kx < Kx; kx++)
						for (int n = nn; n < nn + Tn; n++)
							for (int i = 0; i < Ni; i++) {
								VTYPE sv = synapse[ky][kx][n][i];
								VTYPE nv = neuron_i[ky + y][kx + x][i];
								sum[n] += sv * nv;
							}
				for (int n = nn; n < nn + Tn; n++) {
					neuron_n[yout][xout][n] = transfer(sum[n]);
				}
			}
			xout++;
		}
		yout++;
	}
}

void  convolution_layer(VTYPE(&synapse)[Ky][Kx][Nn][Ni], VTYPE(&neuron_i)[NYPAD][NXPAD][Ni], VTYPE(&neuron_n)[NYSCL][NXSCL][Nn]) {
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
							VTYPE sv = synapse[ky][kx][n][i];
							VTYPE nv = neuron_i[ky + y][kx + x][i];
							sum[n] += sv * nv;
						}
							
				neuron_n[yout][xout][n] = transfer(sum[n]);
			}
			xout++;
		}
		yout++;
	}
}

__global__ void convolution_layer_cuda_2DwithoutTiling(VTYPE* synapse, VTYPE* neuron_i, VTYPE* neuron_n)
{
	VTYPE sum[Nn] = { 0 };
	// — Original code — (excluding nn, ii loops)
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	/*
	* synapse[Ky][Kx][Nn][Ni];
	* neuron_i[NYPAD][NXPAD][Ni];
	* neuron_n[NYSCL][NXSCL][Nn];
	*/

	if (tx < NXSCL && ty < NYSCL) {
		for (int n = 0; n < Nn; n++) {
			sum[n] = 0;

			// x=0
			for (int ky = 0; ky < Ky; ky++)
				for (int kx = 0; kx < Kx; kx++)
					for (int i = 0; i < Ni; i++)
					{
						VTYPE sv = synapse[ky * (Kx * Nn * Ni) + kx * (Nn * Ni) + n * Ni + i];
						VTYPE nv = neuron_i[(ky + ty) * (NXPAD * Ni) + (kx + tx) * Ni + i];

						//synapse[ky][kx][n][i];
						//neuron_i[ky + ty][kx + x][i];

						sum[n] += sv * nv;
					}
			neuron_n[ty * (NXSCL * Nn) + tx * Nn + n] = (sum[n] > 0) ? sum[n] : sum[n] / 4;
			// neuron_n[yout][xout][n] sum[n]
		}
	}
}

__global__ void convolution_layer_cuda_1DwithoutTiling(VTYPE* synapse, VTYPE* neuron_i, VTYPE* neuron_n)
{
	VTYPE sum[Nn] = { 0 };
	// — Original code — (excluding nn, ii loops)
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (int x = 0; x < Ny; x += Sx)
	{ // tiling for x;
		for (int n = 0; n < Nn; n++)
		{
			sum[n] = 0;

			// sliding window;
			for (int ky = 0; ky < Ky; ky++)
				for (int kx = 0; kx < Kx; kx++)
					for (int i = 0; i < Ni; i++)
					{
						VTYPE sv = synapse[ky * (Kx * Nn * Ni) + kx * (Nn * Ni) + n * Ni + i]; //[ky][kx][n][i];
						VTYPE nv = neuron_i[(ky + idx) * (NXPAD * Ni) + (kx + x) * Ni + i]; //[ky + y][kx + x][i];
						sum[n] += sv * nv;
					}

			neuron_n[idx * (NXSCL * Nn) + x * Nn + n] = (sum[n] > 0) ? sum[n] : sum[n] / 4;
		}
	}
}

__global__ void convolution_layer_cuda_2DwithTiling(VTYPE* synapse, VTYPE* neuron_i, VTYPE* neuron_n)
{
	VTYPE sum[Nn] = { 0 };
	// — Original code — (excluding nn, ii loops)
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

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

__global__ void convolution_layer_cuda_1DwithTiling(VTYPE* synapse, VTYPE* neuron_i, VTYPE* neuron_n)
{
	VTYPE sum[Nn] = { 0 };
	// — Original code — (excluding nn, ii loops)
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (int x = 0; x < Ny; x += Sx)
	{ // tiling for x;
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
							VTYPE nv = neuron_i[(ky + idx) * (NXPAD * Ni) + (kx + x) * Ni + i]; //[ky + y][kx + x][i];
							sum[n] += sv * nv;
						}
			for (int n = nn; n < nn + Tn; n++)
			{
				neuron_n[idx * (NXSCL * Nn) + x * Nn + n] = (sum[n] > 0) ? sum[n] : sum[n] / 4;
				//printf("yout: %d, xout: %d, index: %d, neuron:%f\n", y, x, index, neuron_n[y * (NXSCL * Nn) + x * Nn + n]);
			}
		}
	}
}

__global__ void convolution_layer_cuda_3DwithTilingSH(VTYPE* synapse, VTYPE* neuron_i, VTYPE* neuron_n)
{
	VTYPE sum[Nn] = { 0 };
	// — Original code — (excluding nn, ii loops)
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tz = threadIdx.z;
	int block = blockDim.z;

	// __shared__ VTYPE kernel[Ky][Kx][Nn][Ni];

	// for (int nn = 0; nn < Nn; nn += Tn)


	for (int n = tz; n < tz + block; n++)
	{
		sum[n] = 0;
	}

	// sliding window;
	for (int ky = 0; ky < Ky; ky++)
		for (int kx = 0; kx < Kx; kx++)
			for (int n = tz; n < tz + block; n++)
				for (int i = 0; i < Ni; i++)
				{
					VTYPE sv = synapse[ky * (Kx * Nn * Ni) + kx * (Nn * Ni) + n * Ni + i];		//[ky][kx][n][i];
					VTYPE nv = neuron_i[(ky + ty) * (NXPAD * Ni) + (kx + tx) * Ni + i];			//[ky + y][kx + x][i];
					sum[n] += sv * nv;
				}
	for (int n = tz; n < tz + block; n++)
	{
		neuron_n[ty * (NXSCL * Nn) + tx * Nn + n] = (sum[n] > 0) ? sum[n] : sum[n] / 4;
	}
}

int main(const int argc, const char** argv)
{
	cout << "Allocating memory\n";

	synapse = (VTYPE(*)[Ky][Kx][Nn][Ni])malloc(SYNAPSE_SIZE * sizeof(VTYPE));
	neuron_i = (VTYPE(*)[NYPAD][NXPAD][Ni])malloc(NYPAD * NXPAD * Ni * sizeof(VTYPE));
	neuron_n = (VTYPE(*)[NYSCL][NXSCL][Nn])malloc(NYSCL * NXSCL * Nn * sizeof(VTYPE));
	neuron_n2 = (VTYPE(*)[NYSCL][NXSCL][Nn])malloc(NYSCL * NXSCL * Nn * sizeof(VTYPE));
	neuron_n3 = (VTYPE(*)[NYSCL][NXSCL][Nn])malloc(NYSCL * NXSCL * Nn * sizeof(VTYPE));
	neuron_n4 = (VTYPE(*)[NYSCL][NXSCL][Nn])malloc(NYSCL * NXSCL * Nn * sizeof(VTYPE));
	neuron_n5 = (VTYPE(*)[NYSCL][NXSCL][Nn])malloc(NYSCL * NXSCL * Nn * sizeof(VTYPE));

	cout << "Allocating memory for CUDA \n";
	VTYPE* d_synapse = NULL;
	VTYPE* d_neuron_i = NULL;
	VTYPE* d_neuron_n = NULL;

	cudaMalloc((void**)&d_synapse, SYNAPSE_SIZE * sizeof(VTYPE));
	cudaMalloc((void**)&d_neuron_i, NYPAD * NXPAD * Ni * sizeof(VTYPE));
	cudaMalloc((void**)&d_neuron_n, NYSCL * NXSCL * Nn * sizeof(VTYPE));

	cout << "Initializing arrays\n";

	fill_convolution_shared_simple(*synapse, *neuron_i);
	/*
	////DEBUG///////////////////////////////////////////////////////////////////////////////
	for (int n = 0; n < Nn; n++) {
		printf("Output channel %d\n", n);
		for (int i = 0; i < Ni; i++) {
			printf("Input channel %d\n", i);
			for (int y = 0; y < Ky; y++) {
				for (int x = 0; x < Kx; x++) {
					cout << (*synapse)[y][x][n][i];
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

	// 2D threads CUDA version without tiling
	convolution_layer_cuda_2DwithoutTiling<<<dim3((NXSCL+ ThreadsPerBlock -1)/ ThreadsPerBlock, (NXSCL+ ThreadsPerBlock -1)/ ThreadsPerBlock), dim3(ThreadsPerBlock, ThreadsPerBlock) >>>(d_synapse, d_neuron_i, d_neuron_n);
	cudaMemcpy(neuron_n2, d_neuron_n, NYSCL * NXSCL * Nn * sizeof(VTYPE), cudaMemcpyDeviceToHost);
	cout << "2DwithoutTiling: " << cudaGetErrorString(cudaGetLastError()) << endl;
	
	// 1D thread CUDA version without tiling
	convolution_layer_cuda_1DwithoutTiling <<<(NYSCL + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_synapse, d_neuron_i, d_neuron_n);
	cudaMemcpy(neuron_n3, d_neuron_n, NYSCL * NXSCL * Nn * sizeof(VTYPE), cudaMemcpyDeviceToHost);
	cout << "1DwithoutTiling: " << cudaGetErrorString(cudaGetLastError()) << endl;

	// 2D threads CUDA version with tiling
	convolution_layer_cuda_2DwithTiling << <dim3((NXSCL + ThreadsPerBlock - 1) / ThreadsPerBlock, (NXSCL + ThreadsPerBlock - 1) / ThreadsPerBlock), dim3(ThreadsPerBlock, ThreadsPerBlock) >> > (d_synapse, d_neuron_i, d_neuron_n);
	cudaMemcpy(neuron_n4, d_neuron_n, NYSCL * NXSCL * Nn * sizeof(VTYPE), cudaMemcpyDeviceToHost);
	cout << "2DwithTiling: " << cudaGetErrorString(cudaGetLastError()) << endl;


	// 1D thread CUDA version with tiling
	convolution_layer_cuda_1DwithTiling <<<(NYSCL + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock >>>(d_synapse, d_neuron_i, d_neuron_n);
	cudaMemcpy(neuron_n5, d_neuron_n, NYSCL * NXSCL * Nn * sizeof(VTYPE), cudaMemcpyDeviceToHost);
	cout << "1DwithTiling: " << cudaGetErrorString(cudaGetLastError()) << endl;


	// 3D threads CUDA version with tiling 
	convolution_layer_cuda_3DwithTilingSH<<<dim3(BlockPerGrid, BlockPerGrid, BlockPerGrid_z), dim3(ThreadsPerBlock, ThreadsPerBlock, ThreadsPerBlock_z)>>>(d_synapse, d_neuron_i, d_neuron_n);
	cudaMemcpy(neuron_n6, d_neuron_n, NYSCL * NXSCL * Nn * sizeof(VTYPE), cudaMemcpyDeviceToHost);
	cout << "3DwithTiling: " << cudaGetErrorString(cudaGetLastError()) << endl;



	cout << "Cuda versions complete!\n";

	// verify the results
	compare((VTYPE*)*neuron_n, (VTYPE*)*neuron_n2, NYSCL, NXSCL, Nn);
	compare((VTYPE*)*neuron_n, (VTYPE*)*neuron_n3, NYSCL, NXSCL, Nn);
	compare((VTYPE*)*neuron_n, (VTYPE*)*neuron_n4, NYSCL, NXSCL, Nn);
	compare((VTYPE*)*neuron_n, (VTYPE*)*neuron_n5, NYSCL, NXSCL, Nn);
	//compare((VTYPE*)*neuron_n, (VTYPE*)*neuron_n6, NYSCL, NXSCL, Nn);

	free(synapse);
	free(neuron_i);
	free(neuron_n);
	free(neuron_n2);
	free(neuron_n3);
	free(neuron_n4);
	free(neuron_n5);
	free(neuron_n6);
	cudaFree(d_synapse);
	cudaFree(d_neuron_i);
	cudaFree(d_neuron_n);

	cout << "done\n";
}