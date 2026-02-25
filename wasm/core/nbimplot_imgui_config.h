#pragma once

// Enable 32-bit draw indices so large ImGui/ImPlot meshes are not limited
// by 16-bit index buffers.
#define ImDrawIdx unsigned int
