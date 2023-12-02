#pragma once
#include <GLFW/glfw3.h>
#include <iostream>
void initiaLizeGFLW();
void setGLFWWindowHints();
GLFWwindow* createGLFWWindow(int width, int height, const char* windowTitle);
void setupCallbacks(GLFWwindow* window);

