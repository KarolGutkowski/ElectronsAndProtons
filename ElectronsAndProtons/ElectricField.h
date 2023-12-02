#ifndef ELECTRIC_FIELD
#define ELECTRIC_FIELD

#include "cuda_runtime.h"

class ElectricField
{
public:
	int particles_count;
	float2* positions;
	float2* velocities;
	float2* accelerations;
	float2* field;
	int* charges;

	int grid_columns;
	int grid_rows;

	int* particles_grid_cells_d;
	float2* positions_d;
	float2* velocities_d;
	float2* accelerations_d;
	float2* field_d;
	int* charges_d;

	int field_width;
	int field_height;

	int bins_to_check_count;
	int2* bins;
	int2* bins_d;

	ElectricField(int particlesCount, int fieldWidth, int fieldHeight);
	~ElectricField();
private:
	void initializeRandomParticles();
};

#endif