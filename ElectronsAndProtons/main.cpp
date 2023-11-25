#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include "glfw_functions.h"
#include "glad_functions.h"

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600


int main(int argc, char** argv)
{
    initiaLizeGFLW();
    setGLFWWindowHints();
    GLFWwindow* window = createGLFWWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "test");
    glfwMakeContextCurrent(window);
    loadGlad();

    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}