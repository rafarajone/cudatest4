

#include "Renderer.h"

Renderer::Renderer(Window window) : window(window){
	numberOfThreads = window.width;
	numberOfBlocks = window.height;
}



