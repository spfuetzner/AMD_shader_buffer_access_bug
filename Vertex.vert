#version 460
#extension GL_ARB_shader_viewport_layer_array : require

layout (location = 0) out uint subview_idx;

layout (push_constant) uniform SubviewData
{
	uint active_subviews[16];
};

void main()
{
	//#define WORKAROUND //activavate on AMD card to see make it work ==> you shoud rebuild the project after any changes in this shader for them to take effect
	#ifdef WORKAROUND
		for (int i=0;i<16;++i)
			if (i == gl_InstanceIndex)
				subview_idx =  active_subviews[gl_InstanceIndex];
	#else
		subview_idx =  active_subviews[gl_InstanceIndex];
	#endif
	vec2 tex_pos = vec2((gl_VertexIndex << 1) & 2,gl_VertexIndex & 2);
	gl_Position = vec4(tex_pos * 2.0f - 1.0f,0.0f,1.0f);
	gl_ViewportIndex = int(subview_idx);
}

/*

some notes:
this particular problem can be also seen with any buffers / images types provided from pipeline layout (Descriptor Sets). It has nothing to do with multi views or GL_ARB_shader_viewport_layer_array.
This is just simplest way to show the problem.
Especially if both data of interests and index to access this data are from shader interface (outer world) (as can be seen here this also applies to built-ins)
Eg

layout (set = 0,binding = 0) uniform (or buffer) SubviewData
{
	uint active_subviews[16];
};

layout (set = 0, binding = 1) uniform uimage2D AcessData;

It seems there is an issue with that kind of dynamic data access from within the shader.
*/