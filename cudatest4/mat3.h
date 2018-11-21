#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <string>
#include <sstream>

#include "vec3.h"

using namespace std;

struct mat3 {

	float e0, e1, e2, e3, e4, e5, e6, e7, e8;

	__host__ __device__ mat3();
	__host__ __device__ mat3(float, float, float, float, float, float, float, float, float);

	__host__ __device__ friend vec3 operator * (const vec3&, const mat3&);

	string toString();
};