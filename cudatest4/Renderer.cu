
#include "Renderer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

Renderer::Renderer(Window window) : window(window){
	numberOfThreads = window.width;
	numberOfBlocks = window.height;
}

void Renderer::init() {
	const int numberOfBlocks = window.width;
	const int numberOfThreads = window.height;

	const size_t raysS = window.size * 3;
	const size_t raysSB = raysS * sizeof(float);
	cudaMalloc(&d_rays, raysSB);

	void* args[3] = { &window.width, &window.height, &d_rays };
	cudaLaunchKernel<void>(&getRays, numberOfBlocks, numberOfThreads, args);
	
	cudaDeviceSynchronize();

	char_bufferSB = window.size * sizeof(char);
	cudaMalloc(&d_char_buffer, char_bufferSB);
}


void Renderer::render() {
	auto time1 = chrono::high_resolution_clock::now();

	float speed = 0.1f;

	if (GetAsyncKeyState('W')) {
		eye.x += speed * sin(rotation.y);
		eye.z += speed * cos(rotation.y);
	}

	if (GetAsyncKeyState('S')) {
		eye.x += -speed * sin(rotation.y);
		eye.z += -speed * cos(rotation.y);
	}

	if (GetAsyncKeyState('A')) {
		eye.x += -speed * cos(rotation.y);
		eye.z += speed * sin(rotation.y);
	}

	if (GetAsyncKeyState('D')) {
		eye.x += speed * cos(rotation.y);
		eye.z += -speed * sin(rotation.y);
	}

	if (GetAsyncKeyState(VK_SHIFT)) {
		eye.y -= speed;
	}

	if (GetAsyncKeyState(VK_CONTROL)) {
		eye.y += speed;
	}

	if (GetAsyncKeyState(VK_LEFT)) {
		rotation.y -= 0.05f;
	}

	if (GetAsyncKeyState(VK_RIGHT)) {
		rotation.y += 0.05f;
	}


	//raymarch <<<numberOfBlocks, numberOfThreads>>> (eye, rotation, d_rays, d_char_buffer);
	void* args[4] = { &eye, &rotation, &d_rays, &d_char_buffer };
	auto kertime1 = chrono::high_resolution_clock::now();
	cudaLaunchKernel<void>(&raymarch, numberOfBlocks, numberOfThreads, args);
	cudaDeviceSynchronize();
	auto kertime2 = chrono::high_resolution_clock::now();
	string kerdelta = to_string((kertime2 - kertime1).count());

	auto cpytime1 = chrono::high_resolution_clock::now();
	cudaMemcpy(window.char_buffer, d_char_buffer, char_bufferSB, cudaMemcpyDeviceToHost);
	auto cpytime2 = chrono::high_resolution_clock::now();
	string cpydelta = to_string((cpytime2 - cpytime1).count());

	window.drawString(kerdelta, 1, 2);
	window.drawString(cpydelta, 1, 3);

	auto time2 = chrono::high_resolution_clock::now();
	long long delta = (time2 - time1).count();
	AFPS = SECOND / (float)delta;
	this_thread::sleep_for(chrono::nanoseconds(FRAME - delta));
}