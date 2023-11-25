#pragma once
#include <GLAD/glad.h>
#include <iostream>
#include <fstream>

struct ShadersProgramSource
{
	std::string VertexSource;
	std::string FragmentSource;
};

struct ShadersIdentifiers
{
	GLuint vertexShader;
	GLuint fragmentShader;
};

enum class ShaderType
{
	None = -1,
	VertexShader = 0,
	FragmentShader = 1,

};

ShadersProgramSource ParseShader(const std::string& sourceFilePath);
ShadersIdentifiers CompileShaders(ShadersProgramSource& shadersSource);
void CompileShader(GLuint shaderSource, const std::string& shaderName);