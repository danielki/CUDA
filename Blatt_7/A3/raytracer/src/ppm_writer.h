#pragma once

#include "types.h"

void write_ppm(rgb* pixelarray, int width, int height, char* filename);
void write_texture_ppm(rgb* imagePtr, int width, int height, const char* filename, int max);

