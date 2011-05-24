//#define GOOGLE_STRIP_LOG 2
#include <glog/logging.h>
#include <gflags/gflags.h>

#include <fstream>
#include <stdio.h>
#include "types.h"
#include "parser.h"
#include "raytracer.h"
#include "ppm_writer.h"
#include <sys/time.h>
#include "window.h"
#include "antialiasing.h"



static bool validateWidthAndHeight(const char* flagname, int value)
    {
    if (value > 0 && value < 32768)
        return true;
    printf("Invalid value for --%s: %d\n", flagname, (int)value);
    return false;
    }

DEFINE_int32(width, 0, "width of the rendered scene in pixels");
DEFINE_int32(height, 0, "height of the rendered scene in pixels");
static const bool width_dummy = google::RegisterFlagValidator(&FLAGS_width, &validateWidthAndHeight);
static const bool height_dummy = google::RegisterFlagValidator(&FLAGS_height, &validateWidthAndHeight);

int main(int argc, char **argv)
    {
    // logging
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    // flags
    google::ParseCommandLineFlags(&argc, &argv, true);

    // checking command line arguments
   /* if (argc != 3)
        {
        std::cerr << "You have to specify a scene file and a output file" << std::endl;
        return -1;
        }*/

    CHECK_STRNE(argv[1], "") << "No scene file specified.";
    CHECK_STRNE(argv[2], "") << "No output file specified.";

    // parse scene
    scene s;
    parse_scene(argv[1], s);

    // this is our height and width
    int width = FLAGS_width;
    int height = FLAGS_height;

    // this is our final image
    rgb image[height][width];
	
	timeval start;
	timeval end;

	gettimeofday(&start, 0);
        
    // render the scene
    render_image(s, height, width, &image[0][0]);
    
    
	gettimeofday(&end, 0);
	
	double tS = start.tv_sec*1000000 + (start.tv_usec);
    double tE = end.tv_sec*1000000  + (end.tv_usec);
	long diff = long(tE - tS);
    std::cout << "Zeit für das Rendering - Sekunde(n): " <<  diff/1000000
	<< ", Millisekunde(n): " << (diff%1000000)/1000 << "." << std::endl;


	if ( argc == 2 )
	   {
	   openDisplay(height, width, argc, argv, &image[0][0]);
	   }
	else
	   {
	   write_ppm(&image[0][0], width, height, argv[2]);
	   
	   gettimeofday(&start, 0);
	   antialiasing(s, width, height, argv[2]);
	   gettimeofday(&end, 0);
	
	   tS = start.tv_sec*1000000 + (start.tv_usec);
       tE = end.tv_sec*1000000  + (end.tv_usec);
	   diff = long(tE - tS);
       std::cout << "Zeit für das Antialising - Sekunde(n): " 
       <<  diff/1000000 << ", Millisekunde(n): " << (diff%1000000)/1000 << "." << std::endl;
	   }
	

    return 0;
    }
