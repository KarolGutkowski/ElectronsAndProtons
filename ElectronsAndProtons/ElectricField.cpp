#include <time.h>
#include <stdlib.h>
#include "ElectricField.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"



ElectricField::ElectricField(int particlesCount, int fieldWidth, int fieldHeight)
{
	grid_rows = 10;
	grid_columns = 10;

	particles_count = particlesCount;
	field_height = fieldHeight;
	field_width = fieldWidth;

	positions = new float2[particlesCount];
	velocities = new float2[particlesCount];
	accelerations = new float2[particlesCount];
	charges = new int[particlesCount];
	field = new float2[field_width * field_height];

	cudaMalloc((void**)&particles_grid_cells_d, sizeof(int) * particles_count);
	cudaMalloc((void**)&positions_d, sizeof(float2) * particles_count);
	cudaMalloc((void**)&velocities_d, sizeof(float2) * particles_count);
	cudaMalloc((void**)&accelerations_d, sizeof(float2) * particles_count);
	cudaMalloc((void**)&charges_d, sizeof(int) * particles_count);
	cudaMalloc((void**)&field_d, sizeof(float2) * field_height * field_width);


	initializeRandomParticles();
}

ElectricField::~ElectricField()
{
	delete[]positions;
	delete[]velocities;
	delete[]accelerations;
	delete[]charges;
	delete[]field;

	cudaFree(particles_grid_cells_d);
	cudaFree(positions_d);
	cudaFree(velocities_d);
	cudaFree(accelerations_d);
	cudaFree(charges_d);
	cudaFree(field_d);
}

void ElectricField::initializeRandomParticles()
{
	srand(time(NULL));
	float planeWidth = 2; // gl plane is from -1 to 1
	float slowDownFactor = 0.001f;
	for (int i = 0; i < particles_count; i++)
	{
		float2 acceleration;
		acceleration = { 0.0f, 0.0f };

		float2 velocity;
		float randomXVelocity = ((float)rand() / RAND_MAX) * slowDownFactor;
		float randomYVelocity = ((float)rand() / RAND_MAX) * slowDownFactor;
		velocity = { 0.0f, 0.0f };

		float2 position;
		float xPosition = (planeWidth * i) / particles_count - 1.0f;
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

	checkCudaErrors(cudaMemcpy(particles_grid_cells_d, particles_grid_cells, sizeof(int) * particles_count, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(positions_d, positions, sizeof(float2) * particles_count, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(velocities_d, velocities, sizeof(float2) * particles_count, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(accelerations_d, accelerations, sizeof(float2) * particles_count, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(field_d, field, sizeof(float2) * field_width*field_height, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(charges_d, charges, sizeof(int) * particles_count, cudaMemcpyHostToDevice));
}




