
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "vec3.h"

__global__ void getRays(int width, int height, float* a);

__device__ float sphereSDF(const vec3 &p, float r);

__global__ void raymarch(vec3 eye, vec3 rotation, const float* a, char* b);
