
/*
	@author Rafal Rajtar
*/

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
