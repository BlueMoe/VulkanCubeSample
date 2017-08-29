#version 450

layout (location = 0) smooth in vec3 inColor;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inCamera;
layout (location = 3) in vec3 inNormal;

layout (location = 0) out vec4 outFragColor;

layout (binding = 1) uniform sampler2D tex;

void main()
{
	float rim = 1 - clamp(dot(normalize(inNormal),normalize(inCamera)),0,1);
	vec3 color = inColor * 0.5;
	color += rim * vec3(0.17,0.36,0.81) *0.5;
	outFragColor = vec4( color, 0.5 );
	//outFragColor = texture(tex,inUV);
	//outFragColor.a = 0.5;
}