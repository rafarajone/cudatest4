
#include "Renderer.h"

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