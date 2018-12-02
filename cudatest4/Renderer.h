#pragma once

#include <thread>
#include <chrono>

#include "Window.h"
#include "vec3.h"
#include "mat3.h"

struct Renderer {
	Window window;

	const long long SECOND = 1000000000;
	const long long FPS = 60;
	const long long FRAME = SECOND / FPS;

	float AFPS;

	int numberOfThreads, numberOfBlocks;
	
	size_t char_bufferSB;
	char *d_char_buffer;
	char *d_color_buffer;
	float *d_rays;

	vec3 eye = { 0.0f, 0.0f, -5.0f };
	vec3 rotation = { 0.0f, 0.0f, 0.0f };

	Renderer(Window window);

	void init();
	void render();
};