
/*
	@author Rafal Rajtar
*/

#include "kernel.h"

#include <math.h>
#include <Windows.h>
#include <chrono>

#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <thread>

#include "mat3.h"
#include "vec3.h"
#include "Window.h"
#include "Renderer.h"

using namespace std;

__global__ void getRays(int width, int height, float* a) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	int x = i % width;
	int y = i / width;

	float x2 = x / (float)width - 0.5f;
	float y2 = -y / (float)height + 0.5f;

	float length = sqrt(x2 * x2 + y2 * y2 + 1);//TODO: can we optimize this?

	a[i * 3 + 0] = x2 / length;
	a[i * 3 + 1] = y2 / length;
	a[i * 3 + 2] = 1.0f / length;
}

__device__ float sphereSDF(const vec3 &p, float r) {
	return sqrt(p.x * p.x + p.y * p.y + p.z * p.z) - r;
}

__global__ void raymarch(vec3 eye, vec3 rotation, const float* a, char* b) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ int MAX_RAY_STEPS;
	__shared__ float MAX_DIST;
	__shared__ float EPSILON;

	if (threadIdx.x == 0) {
		MAX_RAY_STEPS = 40;
		MAX_DIST = 10000;
		EPSILON = 0.0001f;
	}

	__syncthreads();

	vec3 ray = { a[i * 3 + 0], a[i * 3 + 1] , a[i * 3 + 2] };

	float theta = rotation.y;

	mat3 m(
		cos(theta), 0, sin(theta),
		0, 1, 0,
		-sin(theta), 0, cos(theta)
	);
	ray = ray * m;

	float ray_length = 0;
	for (int j = 0; j < MAX_RAY_STEPS; j++) {
		vec3 p = ray * ray_length + eye;

		p = mod(p + 5.0f, 10.0f) - 5.0f;

		float dist = sphereSDF(p, 1.0);

		if (dist < EPSILON) {
			b[i] = 178;
			return;
		}

		ray_length += dist;

		if (dist > MAX_DIST) break;
	}
	b[i] = ' ';
}

int main() {
	
	Window window(120, 120);
	Renderer renderer(window);
	renderer.init();
	string repdelta;
	while (true) {

		renderer.render();
		window.drawString(to_string(renderer.AFPS), 1, 1);
		window.drawString(repdelta, 1, 4);
		auto reptime1 = chrono::high_resolution_clock::now();
		window.repaint();
		auto reptime2 = chrono::high_resolution_clock::now();
		repdelta = to_string((reptime2 - reptime1).count());
	}

	//system("pause");

    //return 0;
}
