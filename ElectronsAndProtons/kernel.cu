#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"
#include <stdio.h>
#include "helper_cuda.h"
#include <cassert>

#define PARTICLES_COUNT 1000


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

	positions[idx].x += velocities[idx].x * dt;
	positions[idx].y += velocities[idx].y * dt;

	particles[idx * 3] = positions[idx].x;
	particles[idx * 3 + 1] = positions[idx].y;
}


__device__
unsigned char clip(int n)
{
	return n > 255 ? 255 : (n < 0 ? 0 : n);
}


__global__ void intensityKernel(uchar3* grid,const int width,const int height, const float2* positions, const int* charges, const int particlesCount)
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
	int idx = row * width + column;

	for (int i = 0; i < particlesCount; i++)
	{
		float2 pos = positions[i];
		float dx = pixelPosition.x - pos.x;
		float dy = pixelPosition.y - pos.y;
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
}


__host__ void updateField(float* particles, uchar3* grid, ElectricField* field, int particlesCount, float dt, int width, int height)
{
	int blockX = 32;
	int blockY = 32;
	dim3 intensityBlockDim = dim3(blockX, blockY);
	dim3 intensityKernelGridDim = dim3((width + blockX - 1) / blockX, (height + blockY - 1) / blockY);
	intensityKernel<<<intensityKernelGridDim, intensityBlockDim>>>(grid, width, height, field->positions_d, field->charges_d, particlesCount);

	dim3 blockDimensions = dim3(1024);
	dim3 updateKernelGridDimensions = dim3((particlesCount + 1024 - 1) / 1024);
	updateParticlesKernel<<<updateKernelGridDimensions, blockDimensions>>>(particles, field->positions_d, field->velocities_d, particlesCount, dt);
}