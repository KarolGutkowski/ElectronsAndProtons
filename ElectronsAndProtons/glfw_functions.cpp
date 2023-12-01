#include "glfw_functions.h"
#include "interactions.h"

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
	//commented this out cause it disables me from making round points
	/*
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	*/
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

void errorCallback(int error, const char* description) {
	fprintf(stderr, "error %d: %s\n", error, description);
}

void setupCallbacks(GLFWwindow* window)
{
	glfwSetWindowSizeCallback(window, resizeWindowCallback);
	glfwSetErrorCallback(errorCallback);
}
