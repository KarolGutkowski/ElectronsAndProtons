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
	float2* positions_d;
	float2* velocities_d;
	float2* accelerations_d;
	int* charges_d;
	ElectricField(int particlesCount);
	~ElectricField();
private:
	void initializeRandomParticles();
};

#endif