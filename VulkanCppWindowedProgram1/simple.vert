#version 450 core
layout(location = 0) in vec3 inVec;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inNormal;

layout(location = 0) smooth out vec3 outColor;
layout(location = 1) out vec2 outUV;
layout(location = 2) out vec3 outCamera;
layout(location = 3) out vec3 outNormal;

layout(binding = 0) uniform buf {
	mat4 mvpMatrix;
}uniformBuffer;

void main(void)
{
	vec3 camera = vec3(0,3,5);
	outCamera = camera;
	vec4 normal = vec4(inNormal,0);
	normal = uniformBuffer.mvpMatrix*normal;
	outNormal = normal.xyz;
	outUV = inUV;
	outColor = inColor;
	gl_Position = uniformBuffer.mvpMatrix*vec4(inVec,1);
}