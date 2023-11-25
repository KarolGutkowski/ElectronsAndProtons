#include "shader_loader.h"
#include <string>
#include <sstream>

ShadersProgramSource ParseShader(const std::string& sourceFilePath)
{
	std::ifstream stream(sourceFilePath);
	std::string line;

	std::stringstream ss[2];
	ShaderType shaderMode = ShaderType::None;
	while (getline(stream, line))
	{
		if (line.find("#shader") != std::string::npos)
		{
			if (line.find("vertex") != std::string::npos)
			{
				shaderMode = ShaderType::VertexShader;
			}
			else
			{
				shaderMode = ShaderType::FragmentShader;
			}
		}
		else
		{
			ss[(int)shaderMode] << line << '\n';
		}
	}

	return { ss[0].str(), ss[1].str() };
}

ShadersIdentifiers CompileShaders(ShadersProgramSource& shadersSource)
{
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	const char* vertexShaderSource = shadersSource.VertexSource.c_str();
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);

	CompileShader(vertexShader, "VERTEX");

	const char* fragmentShaderSource = shadersSource.FragmentSource.c_str();
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);

	CompileShader(fragmentShader, "FRAGMENT");

	return { vertexShader, fragmentShader };
}

void CompileShader(GLuint shaderSource,const std::string& shaderName)
{
	glCompileShader(shaderSource);

	int success = 0;
	glGetShaderiv(shaderSource, GL_COMPILE_STATUS, &success);

	if (!success)
	{
		char infoLog[512];
		std::cout << "ERROR when compiling shader with name" << shaderName << std::endl;
		std::cout << "Message: " << infoLog << std::endl;
	}
}