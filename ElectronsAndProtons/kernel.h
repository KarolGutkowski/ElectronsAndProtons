#include "ElectricField.h"
#include "SimulationScenarios.h"

void updateField(
	float* particles, 
	uchar3* grid, 
	ElectricField* field, 
	int particlesCount, 
	float dt, int width, 
	int height,
	SimulationScenario scenario);
