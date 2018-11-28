
#include "Window.h"

Window::Window(int width, int height) : width(width), height(height){
	size = width * height;

	char_buffer = new char[size];
	color_buffer = new char[size];

	hOut = GetStdHandle(STD_OUTPUT_HANDLE);

	/*
	CONSOLE_FONT_INFOEX cfi;
	cfi.cbSize = sizeof(cfi);
	cfi.nFont = 0;
	cfi.dwFontSize.X = 8;
	cfi.dwFontSize.Y = 8;
	cfi.FontFamily = FF_DONTCARE;
	cfi.FontWeight = FW_NORMAL;
	wcscpy_s(cfi.FaceName, L"Raster Fonts");
	SetCurrentConsoleFontEx(hOut, false, &cfi);
	*/

	SetConsoleScreenBufferSize(hOut, { (short)width, (short)height });
	SetConsoleActiveScreenBuffer(hOut);
	//SMALL_RECT windowSize = { 0, 0, width, height };
	//SetConsoleWindowInfo(hOut, TRUE, &windowSize);
	SetConsoleTitle("CUDA Raymarching TEST by Rafal Rajtar");
}

void Window::repaint() {
	DWORD dw;
	WriteConsoleOutputCharacter(hOut, char_buffer, size, { 0, 0 }, &dw);
}

void Window::drawString(string text, int x, int y) {
	for (int i = 0; i < text.length(); i++) {
		if (text[i] == '\n') {
			text.erase(0, i);
			i = 0;
			y++;
			continue;
		}
		unsigned int n = x + i + y * width;
		char_buffer[n] = text[i];
		color_buffer[n] = 7;
	}
}