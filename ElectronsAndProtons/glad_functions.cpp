#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include "glad_functions.h"
#include <iostream>

void loadGlad()
{
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initalize GLAD" << std::endl;
        exit(EXIT_SUCCESS);
    }
}