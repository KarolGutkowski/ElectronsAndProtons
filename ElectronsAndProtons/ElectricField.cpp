#include <time.h>
#include <stdlib.h>
#include "ElectricField.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include <iostream>
#include <chrono>


ElectricField::ElectricField(int particlesCount, int fieldWidth, int fieldHeight)
{
	auto start_time = std::chrono::high_resolution_clock::now();
	grid_rows = 40;
	grid_columns = 40;

	particles_count = particlesCount;
	field_height = fieldHeight;
	field_width = fieldWidth;

	positions = new float2[particlesCount];
	velocities = new float2[particlesCount];
	accelerations = new float2[particlesCount];
	charges = new int[particlesCount];
	field = new float2[field_width * field_height];

	bins_to_check_count = 61/*19*/;
	bins = new int2[bins_to_check_count];

	cudaMalloc((void**)&particles_grid_cells_d, sizeof(int) * particles_count);
	cudaMalloc((void**)&positions_d, sizeof(float2) * particles_count);
	cudaMalloc((void**)&velocities_d, sizeof(float2) * particles_count);
	cudaMalloc((void**)&accelerations_d, sizeof(float2) * particles_count);
	cudaMalloc((void**)&charges_d, sizeof(int) * particles_count);
	cudaMalloc((void**)&field_d, sizeof(float2) * field_height * field_width);
	cudaMalloc((void**)&bins_d, sizeof(int2) * bins_to_check_count);


	initializeRandomParticles();
	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
	std::cout << "Data generation time: " << duration.count() << " milliseconds" << std::endl;
}

ElectricField::~ElectricField()
{
	delete[]positions;
	delete[]velocities;
	delete[]accelerations;
	delete[]charges;
	delete[]field;
	delete[]bins;

	cudaFree(particles_grid_cells_d);
	cudaFree(positions_d);
	cudaFree(velocities_d);
	cudaFree(accelerations_d);
	cudaFree(charges_d);
	cudaFree(field_d);
	cudaFree(bins_d);
}

void ElectricField::initializeRandomParticles()
{
	srand(time(NULL));
	float planeWidth = 2; // gl plane is from -1 to 1
	float halfPlaneWidth = 1;
	float slowDownFactor = 0.001f;

	for (int i = 0; i < particles_count; i++)
	{
		float2 acceleration;
		acceleration = { 0.0f, 0.0f };

		float2 velocity;
		float randomXVelocity = ((float)rand() / RAND_MAX) * slowDownFactor;
		float randomYVelocity = ((float)rand() / RAND_MAX) * slowDownFactor;
		velocity = { randomXVelocity, randomYVelocity };

		float2 position;
		float xPosition = (planeWidth * i) / particles_count - halfPlaneWidth;
		float randomYPosition = ((float)rand() / (float)RAND_MAX) * planeWidth - 1.0f;
		position = { xPosition, randomYPosition };

		positions[i] = position;
		velocities[i] = velocity;
		accelerations[i] = acceleration;

		int charge = (rand() / (float)RAND_MAX) * 2;
		charges[i] = charge == 1 ? 1 : -1;
	}


	for (int i = 0; i < field_width * field_height; i++)
	{
		field[i] = { 0.0f, 0.0f };
	}

	int* particles_grid_cells = new int[particles_count];

	for (int i = 0; i < particles_count; i++)
	{
		particles_grid_cells[i] = -1;
	}

	float cutoffDistance = 4.25f;
	int added = 0;
	for (int i = -cutoffDistance; i < cutoffDistance+1; i++)
	{
		for (int j = -cutoffDistance; j < cutoffDistance + 1; j++)
		{
			if (i * i + j * j <= cutoffDistance * cutoffDistance)
			{
				bins[added++] = { i,j };
				//std::cout << "(" << i << "," << j << ")" << std::endl;
			}
		}
	}
	/*if (added == bins_to_check_count) {
		std::cout << "intiialized bins" << std::endl;
	}*/

	auto start_time = std::chrono::high_resolution_clock::now();
	checkCudaErrors(cudaMemcpy(particles_grid_cells_d, particles_grid_cells, sizeof(int) * particles_count, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(positions_d, positions, sizeof(float2) * particles_count, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(velocities_d, velocities, sizeof(float2) * particles_count, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(accelerations_d, accelerations, sizeof(float2) * particles_count, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(field_d, field, sizeof(float2) * field_width*field_height, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(charges_d, charges, sizeof(int) * particles_count, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(bins_d, bins, sizeof(int2) * bins_to_check_count, cudaMemcpyHostToDevice));
	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
	std::cout << "Data copying from CPU-GPU time: " << duration.count()/(float)1000000 << " miliseconds" << std::endl;
}




