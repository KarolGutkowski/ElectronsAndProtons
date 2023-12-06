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
#include "constants.h"
#include "cpu_implementation.h"

void displayMsPerFrame(double& lastTime);

bool run_cpu = false;

int main(int argc, char** argv)
{
    if (argc != 0)
    {
        for (int i = 1; i < argc; i++)
        {
            if (strcmp(argv[i],"-cpu")==0)
            {
                run_cpu = true;
            }
        }
    }

    initiaLizeGFLW();
    setGLFWWindowHints();
    auto window = createGLFWWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Electrons and protons");
    glfwMakeContextCurrent(window);
    loadGlad();
    setupCallbacks(window);

    // group this into a method;
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);


    GLuint program = getProgramFrom("basic_shaders.glsl");

    ElectricField* field = new ElectricField(PARTICLES_COUNT, WINDOW_WIDTH, WINDOW_HEIGHT);

    float* points = (float*)malloc(sizeof(float) * 3 * PARTICLES_COUNT);
    if (!points)
    {
        std::cout << "failed to allocate points" << std::endl;
        return -1;
    }

    memset(points, 0, sizeof(float) * 3 * PARTICLES_COUNT);

    if (!points)
    {
        std::cout << "failed to allocated points array" << std::endl;
        return -1;
    }

    for (int i = 0; i < PARTICLES_COUNT; i++)
    {
        points[3*i] = field->positions[i].x;
        points[3*i +1] = field->positions[i].y;
        points[3 * i + 2] = field->charges[i];
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

    uchar3* cpu_grid = (uchar3*)malloc(sizeof(uchar3) * WINDOW_WIDTH * WINDOW_HEIGHT);

    if (!cpu_grid)
    {
        std::cout << "failed to allocate cpu grid" << std::endl;
        return -1;
    }

    for (int i = 0; i < WINDOW_WIDTH * WINDOW_HEIGHT; i++)
    {
        cpu_grid[i].x = 0;
        cpu_grid[i].y = 0;
        cpu_grid[i].z = 0;
    }

    glBufferData(GL_PIXEL_UNPACK_BUFFER, 3 * WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(GLubyte), cpu_grid, GL_STREAM_DRAW);

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    float* dptr;
    uchar3* grid = 0;
    int W = WINDOW_WIDTH;
    int H = WINDOW_HEIGHT;
    cudaGraphicsResource* cuda_vbo, * cuda_pbo;
    if (!run_cpu)
    {
        cudaGraphicsGLRegisterBuffer(&cuda_vbo, vertices, cudaGraphicsRegisterFlagsWriteDiscard);
        cudaGraphicsMapResources(1, &cuda_vbo);

        cudaGraphicsResourceGetMappedPointer((void**)&dptr, NULL, cuda_vbo);

        cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
        cudaGraphicsMapResources(1, &cuda_pbo);

        cudaGraphicsResourceGetMappedPointer((void**)&grid, NULL, cuda_pbo);
    }
    double lastTime = glfwGetTime();
    int nbFrames = 0;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        float dt = 0.1f;
        if (!run_cpu)
            updateField(dptr, grid, field, PARTICLES_COUNT, dt, W, H);
        else
        {
            updateField(field, points, W, H, PARTICLES_COUNT, dt, cpu_grid);
            glBufferData(GL_PIXEL_UNPACK_BUFFER, 3 * WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(GLubyte), cpu_grid, GL_STREAM_DRAW);
        }

        
        glClear(GL_COLOR_BUFFER_BIT);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0, -1.0);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0, 1.0);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0, 1.0);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0, -1.0);
        glEnd();
        glDisable(GL_TEXTURE_2D);

        glUseProgram(program);

        glBindVertexArray(VAO);

        if (run_cpu)
        {
            glBindBuffer(GL_ARRAY_BUFFER, vertices);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * PARTICLES_COUNT * 3, points, GL_DYNAMIC_DRAW);
        }
        glDrawArrays(GL_POINTS, 0, PARTICLES_COUNT);

        glBindVertexArray(0);
        glUseProgram(0);

        glfwSwapBuffers(window);
        displayMsPerFrame(lastTime);
    }

    if (!run_cpu)
    {
        cudaGraphicsUnmapResources(1, &cuda_vbo);
        cudaGraphicsUnmapResources(1, &cuda_pbo);
    }

    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    delete field;

    glfwTerminate();
    return 0;
}


int nbFrames = 0;
void displayMsPerFrame(double& lastTime)
{
    double currentTime = glfwGetTime();
    nbFrames++;
    if (currentTime - lastTime >= 1.0) {
        double elapsed = currentTime - lastTime;
        auto timePerFrame = elapsed* 1000.0 / double(nbFrames);
        auto framesPerSecond = 1000.0 / timePerFrame;
        printf("%f ms/frame (fps=%f)\n", timePerFrame, framesPerSecond);
        nbFrames = 0;
        lastTime += 1.0;
    }
}