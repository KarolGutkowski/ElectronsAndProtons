#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include "glfw_functions.h"
#include "glad_functions.h"
#include "program_loader.h"
#include "kernel.h"
#include "cuda_gl_interop.h"
#include "ElectricField.h"
#include <cassert>

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define PARTICLES_COUNT 1000

int main(int argc, char** argv)
{
    initiaLizeGFLW();
    setGLFWWindowHints();
    auto window = createGLFWWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "test");
    glfwMakeContextCurrent(window);
    loadGlad();
    setupCallbacks(window);

    // group this into a method;
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    //

    GLuint program = getProgramFrom("basic_shaders.glsl");

    ElectricField* field = new ElectricField(PARTICLES_COUNT);

    float* points = (float*)malloc(sizeof(float) * 3 * PARTICLES_COUNT);

    if (!points)
    {
        std::cout << "failed to allocated points array" << std::endl;
        return -1;
    }

    for (int i = 0; i < PARTICLES_COUNT; i++)
    {
        points[3*i] = field->positions[i].x;
        points[3*i +1] = field->positions[i].y;
        points[3 * i + 2] = 0.0f;
    }

    
    /*for (int i = 0; i < PARTICLES_COUNT; i++)
    {
        std::cout << "x= " << points[3 * i] << " ";
        std::cout << "y= " << points[3 * i + 1] << " ";
        std::cout << "z= " << points[3 * i + 2] << std::endl;
    }*/
    

    GLuint vertices;
    glGenBuffers(1, &vertices);
    glBindBuffer(GL_ARRAY_BUFFER, vertices);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * PARTICLES_COUNT*3, points, GL_DYNAMIC_DRAW);

    GLuint VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, vertices);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    GLuint pbo = 0;;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(GLubyte), 0, GL_STREAM_DRAW);

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINE);


    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        cudaGraphicsResource* cuda_vbo, *cuda_pbo;

        cudaGraphicsGLRegisterBuffer(&cuda_vbo, vertices, cudaGraphicsRegisterFlagsWriteDiscard);
        cudaGraphicsMapResources(1, &cuda_vbo);

        float* dptr;
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cuda_vbo);

        cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
        cudaGraphicsMapResources(1, &cuda_pbo);

        uchar4* grid = 0;
        size_t grid_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&grid, &grid_bytes, cuda_pbo);

        int W = WINDOW_WIDTH;
        int H = WINDOW_HEIGHT;
        updateField(dptr, grid, field, PARTICLES_COUNT, 10.0f, W, H);

        cudaGraphicsUnmapResources(1, &cuda_vbo);
        cudaGraphicsUnmapResources(1, &cuda_pbo);

        glClear(GL_COLOR_BUFFER_BIT);
        //glClearColor(0.18, 0.35, 0.53, 1.0);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0,-1.0);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0, 1.0);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0, 1.0);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0, -1.0);
        glEnd();
        glDisable(GL_TEXTURE_2D);

        glUseProgram(program);
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, PARTICLES_COUNT);
        glUseProgram(0);

        glfwSwapBuffers(window);
    }

    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    delete field;

    glfwTerminate();
    return 0;
}