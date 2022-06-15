#version 460

layout (location = 0) in flat uint subview_idx;
layout (location = 0) out vec4 color;

void main()
{
	// if node id is invalid then set red color to the created bounding boxes
	if (subview_idx == 0) // remove the if and you ll get crash with official driver releases since 397.31 (newest to date tested either)
	{
		color = vec4(1,0,0,1);
	}
	else
	{
		color = vec4(0,1,0,1);
	}
}