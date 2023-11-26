#ifndef ELECTRIC_FIELD
#define ELECTRIC_FIELD

#include "cuda_runtime.h"

class ElectricField
{
public:
	int particlesCount;
	float2* positions;
	float2* velocities;
	float2* accelerations;
	int* charges;
	ElectricField(int particlesCount);
	~ElectricField();
private:
	void initializeRandomParticles();
};

#endif