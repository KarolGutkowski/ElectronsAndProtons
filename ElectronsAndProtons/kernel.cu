#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"
#include <stdio.h>
#include "helper_cuda.h"
#include <cassert>

__global__ void updateParticlesKernel(float* particles, float2* positions, float2* velocities, const int particlesCount, float dt)
{
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= particlesCount)
		return;

	if (positions[idx].x >= 1.0f || positions[idx].x <= -1.0f)
	{
		velocities[idx].x *= -1;
	}

	if (positions[idx].y >= 1.0f || positions[idx].y <= -1.0f)
	{
		velocities[idx].y *= -1;
	}

	positions[idx].x += velocities[idx].x * dt * 0.001f;
	positions[idx].y += velocities[idx].y * dt * 0.001f;

	particles[idx * 3] = positions[idx].x;
	particles[idx * 3 + 1] = positions[idx].y;


	//printf("x= %f, y= %f (v=(%f,%f))\n", positions[idx].x, positions[idx].y, velocities[idx].x, velocities[idx].y);
}


__device__
unsigned char clip(int n)
{
	return n > 255 ? 255 : (n < 0 ? 0 : n);
}


__global__ void intensityKernel(uchar4* grid,const int width,const int height, const float2* positions, const int* charges, const int particlesCount)
{
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (column >= width || row >= height)
		return;

	float aspectRatio = width / height;

	float2 pixelPosition =
	{
		(column / (float)(width - 1) * 2.0f - 1.0f)* aspectRatio,
		(row / (float)(height - 1) * 2.0f - 1.0f)
	};

	float intensity = 0.0f;
	int count = particlesCount;
	int idx = row * width + column;

	for (int i = 0; i < count; i++)
	{
		float2 pos = positions[i];
		float dx = pixelPosition.x - pos.x;
		float dy = pixelPosition.y - pos.y;

		//printf("charge[%d]= $f , pixelPosition = (%f,%f), particle[%d] = (%f,%f) dx = %f, dy = %f \n", i, charge, pixelPosition.x, pixelPosition.y, i, pos.x, pos.y, dx, dy);

		intensity += charges[i] / sqrtf(dx * dx + dy * dy);
	}
	
	if (intensity < 0)
	{
		intensity *= -10;
		grid[idx].z = clip(intensity);
		grid[idx].x = 0;
	}
	else
	{
		intensity *= 10;
		grid[idx].x = clip(intensity);
		grid[idx].z = 0;
	}
	grid[idx].w = 255;
}



__host__ void updateField(float* particles, uchar4* grid, ElectricField* field, int particlesCount, float dt, int width, int height)
{
	float2* postions_d;
	cudaMalloc((void**)&postions_d, sizeof(float2) * particlesCount);
	checkCudaErrors(cudaMemcpy(postions_d, field->positions, sizeof(float2) * particlesCount, cudaMemcpyHostToDevice));

	float2* velocities_d;
	cudaMalloc((void**)&velocities_d, sizeof(float2) * particlesCount);
	checkCudaErrors(cudaMemcpy(velocities_d, field->velocities, sizeof(float2) * particlesCount, cudaMemcpyHostToDevice));

	int* charges_d;
	cudaMalloc((void**)&charges_d, sizeof(int) * particlesCount);
	checkCudaErrors(cudaMemcpy(charges_d, field->charges, sizeof(int) * particlesCount, cudaMemcpyHostToDevice));

	dim3 blockDimensions = dim3(1024);
	dim3 updateKernelGridDimensions = dim3((particlesCount + 1024 - 1) / 1024);
	updateParticlesKernel<<<updateKernelGridDimensions, blockDimensions >>>(particles, postions_d, velocities_d, particlesCount, dt);

	dim3 intensityBlockDim = dim3(32, 32);
	dim3 intensityKernelGridDim = dim3((width + 32 - 1) / 32, (height + 32 - 1) / 32);

	//printf("width = %d, height = %d\n", width, height);
	//printf(" intensityKernelGridDim = dim3(%d, %d)\n", intensityKernelGridDim.x, intensityKernelGridDim.y);
	intensityKernel<<<intensityKernelGridDim, intensityBlockDim >>> (grid, width, height, postions_d, charges_d, particlesCount);

	checkCudaErrors(cudaMemcpy(field->positions, postions_d, sizeof(float2) * particlesCount, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(field->velocities, velocities_d, sizeof(float2) * particlesCount, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(field->charges, charges_d, sizeof(int) * particlesCount, cudaMemcpyDeviceToHost));

	cudaFree(postions_d);
	cudaFree(velocities_d);
	cudaFree(charges_d);

	cudaDeviceSynchronize();
}