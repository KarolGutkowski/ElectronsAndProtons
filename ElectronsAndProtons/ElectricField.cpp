#include <time.h>
#include <stdlib.h>
#include "ElectricField.h"


ElectricField::ElectricField(int particlesCount)
{
	this->particlesCount = particlesCount;
	this->positions = new float2[particlesCount];
	this->velocities = new float2[particlesCount];
	this->accelerations = new float2[particlesCount];
	this->charges = new int[particlesCount];
	initializeRandomParticles();
}

ElectricField::~ElectricField()
{
	delete[]positions;
	delete[]velocities;
	delete[]accelerations;
}

void ElectricField::initializeRandomParticles()
{
	srand(time(NULL));
	float planeWidth = 2; // gl plane is from -1 to 1
	float slowDownFactor = 0.8f;
	for (int i = 0; i < particlesCount; i++)
	{
		float2 acceleration;
		acceleration = { 0.0f, 0.0f };

		float2 velocity;
		float randomXVelocity = ((float)rand() / RAND_MAX )* slowDownFactor;
		float randomYVelocity = ((float)rand() / RAND_MAX) * slowDownFactor;
		velocity = { randomXVelocity, randomYVelocity };


		float2 position;
		float xPosition = (planeWidth * i) / particlesCount - 1.0f;
		float randomYPosition = ((float)rand() /(float)RAND_MAX)*planeWidth - 1.0f;
		position = { xPosition, randomYPosition };

		positions[i] = position;
		velocities[i] = velocity;
		accelerations[i] = acceleration;

		int charge = (rand() /(float) RAND_MAX) * 2;
		charges[i] = charge == 1? 1 : -1; // -1 negative, 1 positive
	}
}




