#include "ppm_writer.h"
#include <iostream>
#include <fstream>

void write_ppm(rgb* pixelarray, int width, int height, char* filename)
    {
    std::ofstream f(filename);
    if(!f) return; // kann nicht geöffnent werden
    f << "P3" << std::endl;
    f << width << " " << height << std::endl;
    f << "255" << std::endl;
    for(int h=0; h < height; h++)
        {
        for(int w=0; w < width; w++)
            {
            f << pixelarray[h*width+w] << std::endl;
            }
        }
    f.close();
    }
	
void write_texture_ppm(rgb* imagePtr, int width, int height, const char* filename, int max) 
	{
	std::ofstream f(filename);
	f << "P3\n";
	f << width << " " << height << "\n";
	f << max << "\n";

	for (int i = 0; i < width * height; ++i) 
		{
		f << static_cast<int>(imagePtr[i].x) << " ";
		f << static_cast<int>(imagePtr[i].y) << " ";
		f << static_cast<int>(imagePtr[i].z) << "\n";
		}	
	}

