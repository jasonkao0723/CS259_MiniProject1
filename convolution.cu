#include <iostream>
#include <string>
#include <stdio.h>
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
#define Ti 16

#define Ty 8
#define Tx 8
#endif

#define NYPAD (Ny + Ky)
#define NXPAD (Nx + Kx)

#define NYSCL (Ny / Sy)
#define NXSCL (Nx / Sx)

#define SYNAPSE_SIZE (1L * Ky * Kx * Nn * Ni)

VTYPE (*synapse)
[Ky][Kx][Nn][Ni];

VTYPE (*neuron_i)
[NYPAD][NXPAD][Ni];
VTYPE (*neuron_n)
[NYSCL][NXSCL][Nn];
VTYPE (*neuron_n2)
[NYSCL][NXSCL][Nn];

void* aligned_malloc(uint64_t align, uint64_t bytes)  {
	size_t mask = (align-1)^((size_t)-1);
	char* ptr = (((char*)malloc(bytes+align)) + align);
	ptr = (char*) (((size_t)ptr) & mask);
	return (void*) ptr;
}
  

void fill_convolution_shared_simple(VTYPE (&synapse)[Ky][Kx][Nn][Ni],
									VTYPE (&neuron_i)[NYPAD][NXPAD][Ni])
{
	for (int yy = 0; yy < Ky; ++yy)
	{
		for (int xx = 0; xx < Kx; ++xx)
		{
			for (int nn = 0; nn < Nn; ++nn)
			{
				for (int ni = 0; ni < Ni; ++ni)
				{
					synapse[yy][xx][nn][ni] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
				}
			}
		}
	}
	for (int yy = 0; yy < NYPAD; ++yy)
	{
		for (int xx = 0; xx < NXPAD; ++xx)
		{
			for (int ni = 0; ni < Ni; ++ni)
			{
				neuron_i[yy][xx][ni] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
			}
		}
	}
}

void convolution_layer_blocked(
	VTYPE (&synapse)[Ky][Kx][Nn][Ni],
	VTYPE (&neuron_i)[NYPAD][NXPAD][Ni],
	VTYPE (&neuron_n)[NYSCL][NXSCL][Nn])
{
	int c1 = 0, c2 = 0;
	VTYPE sum[Nn] = {0};

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

void convolution_layer(VTYPE (&synapse)[Ky][Kx][Nn][Ni],
					   VTYPE (&neuron_i)[NYPAD][NXPAD][Ni],
					   VTYPE (&neuron_n)[NYSCL][NXSCL][Nn])
{
	VTYPE sum[Nn] = {0};

	// — Original code — (excluding nn, ii loops)
	int yout = 0;
	for (int y = 0; y < Ny; y += Sy)
	{ // tiling for y;
		int xout = 0;
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
								VTYPE sv = synapse[ky][kx][n][i];
								VTYPE nv = neuron_i[ky + y][kx + x][i];
								sum[n] += sv * nv;
							}
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

__global__ void conv_layer_exp1(VTYPE *synapse, VTYPE *neuron_i, VTYPE *neuron_n) {
	int i = blockIdx.x;
	int j = threadIdx.x;
	int stride_i = gridDim.x;
	int stride_j = blockDim.x;
	VTYPE sum[Nn] = {0};
	for (int y = i; y < Ny; y+=stride_i) {
		for (int x = j; x < Nx; x+=stride_j) {
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
								VTYPE nv = neuron_i[(ky + y) * (NXPAD * Ni) + (kx + x) * Ni + i]; //[ky + y][kx + x][i];
								sum[n] += sv * nv;
							}
				for (int n = nn; n < nn + Tn; n++)
				{
					neuron_n[y * (NXSCL * Nn) + x * Nn + n] = (sum[n]>0) ? sum[n] : sum[n]/4;
				}
			}
		}
	}
}

int main(const int argc, const char **argv)
{
	cout << "allocating memory\n";

	// Comment the following if you want to experiment with aligned_malloc
	// synapse   = (VTYPE (*)[Ky][Kx][Nn][Ni])  aligned_malloc(64,  SYNAPSE_SIZE*sizeof(VTYPE));
  	// neuron_i  = (VTYPE (*)[NYPAD][NXPAD][Ni])aligned_malloc(64,NYPAD*NXPAD*Ni*sizeof(VTYPE));
  	// neuron_n  = (VTYPE (*)[NYSCL][NXSCL][Nn])aligned_malloc(64,NYSCL*NXSCL*Nn*sizeof(VTYPE));
  	// neuron_n2 = (VTYPE (*)[NYSCL][NXSCL][Nn])aligned_malloc(64,NYSCL*NXSCL*Nn*sizeof(VTYPE));
	
	synapse = (VTYPE(*)[Ky][Kx][Nn][Ni])malloc(SYNAPSE_SIZE * sizeof(VTYPE));
	neuron_i = (VTYPE(*)[NYPAD][NXPAD][Ni])malloc(NYPAD * NXPAD * Ni * sizeof(VTYPE));
	neuron_n = (VTYPE(*)[NYSCL][NXSCL][Nn])malloc(NYSCL * NXSCL * Nn * sizeof(VTYPE));
	neuron_n2 = (VTYPE(*)[NYSCL][NXSCL][Nn])malloc(NYSCL * NXSCL * Nn * sizeof(VTYPE));

	cout << "allocating memory for CUDA \n";
	VTYPE* d_synapse = NULL;
	VTYPE* d_neuron_i = NULL;
	VTYPE* d_neuron_n = NULL;
	VTYPE* d_neuron_n2 = NULL;

	cudaMalloc((void**) &d_synapse, SYNAPSE_SIZE * sizeof(VTYPE));
	cudaMalloc((void**) &d_neuron_i, NYPAD * NXPAD * Ni * sizeof(VTYPE));
	cudaMalloc((void**) &d_neuron_n, NYSCL * NXSCL * Nn * sizeof(VTYPE));
	// reserved for blocked version of the conv layer
	cudaMalloc((void**) &d_neuron_n2, NYSCL * NXSCL * Nn * sizeof(VTYPE));

	cout << "initializing arrays\n";

	fill_convolution_shared_simple(*synapse, *neuron_i);

	cout << "copying initialized arrays from host to device\n";

	cudaMemcpy(d_synapse, synapse, SYNAPSE_SIZE * sizeof(VTYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(d_neuron_i, neuron_i, NYPAD * NXPAD * Ni * sizeof(VTYPE), cudaMemcpyHostToDevice);

	cout << "starting computation\n";

	// simple Version
	convolution_layer(*synapse, *neuron_i, *neuron_n);
	cout << "simple version complete!\n";

	// simple CUDA version
	conv_layer_exp1<<<128,128>>>(d_synapse, d_neuron_i, d_neuron_n);
	cudaDeviceSynchronize();
	cudaMemcpy(neuron_n2, d_neuron_n, NYSCL * NXSCL * Nn * sizeof(VTYPE), cudaMemcpyDeviceToHost);
	cout << "cuda simple version complete!\n";

	// // //Blocked Version
	// // convolution_layer_blocked(*synapse, *neuron_i, *neuron_n2);
	// // cout << "blocked computation complete!\n";

	// verify the results
	compare((VTYPE *)*neuron_n, (VTYPE *)*neuron_n2, NYSCL * NXSCL * Nn);

	free(synapse);
	free(neuron_i);
	free(neuron_n);
	free(neuron_n2);
	cudaFree(d_synapse);
	cudaFree(d_neuron_i);
	cudaFree(d_neuron_n);
	cudaFree(d_neuron_n2);
	cout << "done\n";
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// compile flags: nvcc convolution.cu -o conv -DNx=224 -DNy=224 -DKx=3 -DKy=3 -DNi=64 -DNn=64 -DTii=32 -DTi=16 -DTnn=32 -DTn=16 -DTx=7 -DTy=7
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////