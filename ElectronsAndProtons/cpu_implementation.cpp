#include "cpu_implementation.h"
#include <iostream>

void calculate_intensity(float2* field, uchar3* grid, int width, int height, float* positions, int* charges, int particles_count);
void update_particles(float* positions, float2* field, int* charges, int particles_count, int width, int height, float2* velocities, float2* accelerations, float dt);

///
// field - for calculation purposes
// positions - calculations but also gl display
// grid - purely opengl display
// charges - for calculation purposes
///
void updateField(ElectricField* field,float* positions, int width, int height, int particles_count, float dt, uchar3* grid)
{
	calculate_intensity(field->field, grid, width, height, positions, field->charges, particles_count);
	update_particles(positions, field->field, field->charges, particles_count, width, height, field->velocities, field->accelerations, dt);
}

char clip(int n)
{
	return n > 255 ? 255 : (n < 0 ? 0 : n);
}

void calculate_intensity(float2* field, uchar3* grid, int width, int height, float* positions, int* charges, int particles_count)
{
	int pixel_count = width * height;
	float aspectRatio = width / height;

	for (int j = 0; j < height; j++)
	{
		for (int k = 0; k < width; k++)
		{
			float total_intensity = 0.0f;
			for (int i = 0; i < particles_count; i++)
			{
				float x = positions[3 * i];
				float y = positions[3 * i + 1];

				float pixelPositionX = (k / (float)(width - 1) * 2.0f - 1.0f) * aspectRatio;
				float pixelsPositionY = (j / (float)(height - 1) * 2.0f - 1.0f);

				float dx = pixelPositionX - x;
				float dy = pixelsPositionY - y;

				float distance = sqrtf(dx * dx + dy * dy);
				float intensity = charges[i] / distance;

				if (distance > 0.01f)
				{
					field[j * width + k].x += intensity * (dx / distance)/100.0f;
					field[j * width + k].y += intensity * (dx / distance) / 100.0f;
				}

				total_intensity += intensity;
			}
			if (total_intensity < 0)
			{
				total_intensity *= -1;
				grid[j * width + k].z = clip(total_intensity);
				grid[j * width + k].x = 0;
			}
			else
			{
				grid[j * width + k].x = clip(total_intensity);
				grid[j * width + k].z = 0;
			}
		}
	}
}


void update_particles(float* positions, float2* field, int* charges, int particles_count, int width, int height, float2* velocities, float2* accelerations, float dt)
{
	for (int i = 0; i < particles_count; i++)
	{
		int charge = charges[i];
		float positionX = positions[3 * i];
		float positionY = positions[3 * i + 1];

		float2 velocity = velocities[i];

		if (positionX >= 1.0f || positionX <= -1.0f)
		{
			velocity.x *= -1;
		}

		if (positionY >= 1.0f || positionY <= -1.0f)
		{
			velocity.y *= -1;
		}

		positions[3 * i] = positionX + velocity.x * dt;
		positions[3 * i + 1] = positionY + velocity.y * dt;

		if (positions[3 * i + 1] > 1.0f)
		{
			positions[3 * i + 1] = 1.0f;
		}
		else if (positions[3 * i + 1] < -1.0f)
		{
			positions[3 * i + 1] = -1.0f;
		}

		if (positions[3 * i] > 1.0f)
		{
			positions[3 * i] = 1.0f;
		}
		else if (positions[3 * i] < -1.0f)
		{
			positions[3 * i] = -1.0f;
		}

		int2 pixel =
		{
			(int)roundf((positionX + 1.0f) * (width - 1) / 2.0f),
			(int)roundf((positionY + 1.0f) * (height - 1) / 2.0f)
		};


		int positionIdx = pixel.x + pixel.y * width;
		if (pixel.x >= width || pixel.y >= height || positionIdx >= width * height)
		{
			std::cout << "ERROR: particle[ " << i << "] is outside of the screen" << std::endl;
			continue;
		}

		float mass = charge > 0 ? 18360 : 1000;

		accelerations[i].x = charge * field[positionIdx].x / mass;
		accelerations[i].y = charge * field[positionIdx].y / mass;

		velocities[i].x = velocity.x + accelerations[i].x * dt;
		velocities[i].y = velocity.y + accelerations[i].y * dt;
	}
}