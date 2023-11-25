#include "glad/glad.h"
#include "interactions.h"

void resizeWindowCallback(GLFWwindow* window, int width, int height)
{
	glfwSetWindowSize(window, width, height);
	glViewport(0, 0, width, height);
}