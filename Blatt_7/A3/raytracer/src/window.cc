#include <glog/logging.h>
#include <GL/gl.h>		   // Open Graphics Library (OpenGL) header
#include <GL/glut.h>	   // The GL Utility Toolkit (GLUT) Header
#include "types.h"


int width;
int height;
float* image;

void drawImage()
    {
    glDrawPixels(width, height, GL_RGB, GL_FLOAT, image );
    glutSwapBuffers();
    }

void openDisplay(int h, int w, int argc, char **argv, rgb* pixelarray)
    {
    width = w;
    height = h;
    
    image = (float*)malloc(width*height*3*sizeof(float));
    
    // Konvertierung 
    for ( int y=0; y < height; y++ )
        {
        for ( int x=0; x < width; x++ )
            {
            // Standartisierung mit Float und / 255.0
            image[3*(y*width + x)    ]=pixelarray[(height-1-y)*width + x].x / 255.0;
            image[3*(y*width + x) + 1]=pixelarray[(height-1-y)*width + x].y / 255.0;
            image[3*(y*width + x) + 2]=pixelarray[(height-1-y)*width + x].z / 255.0;
            }
        }
    
    glutInit(&argc, argv);                                      
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH );  
	glutInitWindowPosition(1,1);
	glutInitWindowSize(width,height);
	glutCreateWindow("Scene");		
	glutDisplayFunc(drawImage); 
	glutMainLoop(); 
	}
	
	
