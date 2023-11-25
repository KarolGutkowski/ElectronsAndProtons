#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include "glfw_functions.h"
#include "glad_functions.h"
#include "shader_loader.h"

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600


int main(int argc, char** argv)
{
    initiaLizeGFLW();
    setGLFWWindowHints();
    auto window = createGLFWWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "test");
    glfwMakeContextCurrent(window);
    loadGlad();
    setupCallbacks(window);

    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    auto shadersSource = ParseShader("basic_shader.glsl");
    auto shaders = CompileShaders(shadersSource);

    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}