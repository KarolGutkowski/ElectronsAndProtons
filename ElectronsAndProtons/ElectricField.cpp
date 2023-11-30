#include <time.h>
#include <stdlib.h>
#include "ElectricField.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"



ElectricField::ElectricField(int particlesCount)
{
	this->particlesCount = particlesCount;
	positions = new float2[particlesCount];
	velocities = new float2[particlesCount];
	accelerations = new float2[particlesCount];
	charges = new int[particlesCount];

	cudaMalloc((void**)&positions_d, sizeof(float2) * particlesCount);
	cudaMalloc((void**)&velocities_d, sizeof(float2) * particlesCount);
	cudaMalloc((void**)&charges_d, sizeof(int) * particlesCount);

	initializeRandomParticles();
}

ElectricField::~ElectricField()
{
	delete[]positions;
	delete[]velocities;
	delete[]accelerations;

	cudaFree(positions_d);
	cudaFree(velocities_d);
	cudaFree(charges_d);
}

void ElectricField::initializeRandomParticles()
{
	srand(time(NULL));
	float planeWidth = 2; // gl plane is from -1 to 1
	float slowDownFactor = 0.001f;
	for (int i = 0; i < particlesCount; i++)
	{
		float2 acceleration;
		acceleration = { 0.0f, 0.0f };

		float2 velocity;
		float randomXVelocity = ((float)rand() / RAND_MAX) * slowDownFactor;
		float randomYVelocity = ((float)rand() / RAND_MAX) * slowDownFactor;
		velocity = { randomXVelocity, randomYVelocity };

		float2 position;
		float xPosition = (planeWidth * i) / particlesCount - 1.0f;
		float randomYPosition = ((float)rand() / (float)RAND_MAX) * planeWidth - 1.0f;
		position = { xPosition, randomYPosition };

		positions[i] = position;
		velocities[i] = velocity;
		accelerations[i] = acceleration;

		int charge = (rand() / (float)RAND_MAX) * 2;
		charges[i] = charge == 1 ? 1 : -1;
	}

	checkCudaErrors(cudaMemcpy(positions_d, positions, sizeof(float2) * particlesCount, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(velocities_d, velocities, sizeof(float2) * particlesCount, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(charges_d, charges, sizeof(int) * particlesCount, cudaMemcpyHostToDevice));
}




