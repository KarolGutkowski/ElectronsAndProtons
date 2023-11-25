#shader vertex
#version 330 core
layout (location = 0) in vec3 position;

void main()
{
	gl_Position = vec4(position, 0.0);
}

#shader fragment
#version 330 core

out vec3 fragmentColor;

void main()
{
	fragmentColor = vec4(0.4, 0.3, 0.6, 1.0);
}