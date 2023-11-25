#include "glfw_functions.h"

void initiaLizeGFLW()
{
	if (!glfwInit())
	{
		std::cout << "Error when trying to initialize glfw" << std::endl;
		exit(EXIT_FAILURE);
	}
}

void setGLFWWindowHints()
{
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
}

GLFWwindow* createGLFWWindow(int width, int height,const char* windowTitle)
{
	GLFWwindow* window;
	window = glfwCreateWindow(width, height, windowTitle, NULL, NULL);

	if (!window)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	return window;
}
