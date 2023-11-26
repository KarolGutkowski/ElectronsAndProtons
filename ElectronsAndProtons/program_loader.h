#pragma once
#include <GLFW/glfw3.h>
#include <iostream>
#include "shader_loader.h"

GLuint getProgramFrom(const std::string& shadersSourceFilePath)
{
	auto shadersSource = ParseShader(shadersSourceFilePath);
	auto shaders = CompileShaders(shadersSource);

    GLuint program = glCreateProgram();
    glAttachShader(program, shaders.vertexShader);
    glAttachShader(program, shaders.fragmentShader);

    glLinkProgram(program);

    int success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success)
    {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cout << "error linking program " << infoLog << std::endl;
    }
    glDeleteShader(shaders.vertexShader);
    glDeleteShader(shaders.fragmentShader);

    return program;
}
