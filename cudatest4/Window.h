#pragma once

#include <Windows.h>
#include <iostream>
#include <chrono>
#include <thread>

using namespace std;

class Window {
public:
	SHORT width, height;
	int size;

	char* char_buffer;
	char* color_buffer;

	HANDLE hOut;

	Window(SHORT width, SHORT height);
	
	void repaint();
	void drawString(string text, int x, int y);

};