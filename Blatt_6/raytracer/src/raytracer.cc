#include <glog/logging.h>
#include <cmath>
#include <stdio.h>
#include "raytracer.h"
#include "types.h"
#include "float.h"

struct  deviceData
    {
    point stdRechts;
    point stdRunter;
	rgb hintergrund;
    camera cam;
	int height;
	int width;
	int primSize;
	int lightSize;
    };


 __device__ __host__ point cross(const point &p1, const point &p2)
    {
    point point;
    point.x = p1.y*p2.z - p1.z*p2.y;
    point.y = p1.z*p2.x - p1.x*p2.z;
    point.z = p1.x*p2.y - p1.y*p2.x;
    return point;
    }

 __device__ __host__ float dot(const point &p1, const point &p2)
    {
    float f;
    f = p1.x*p2.x+p1.y*p2.y+p1.z*p2.z;
    return f;
    }

 __device__ __host__ float norm(const point &p)
    {
    float f;
    f = sqrt((p.x*p.x)+(p.y*p.y)+(p.z*p.z));
    return f;
    }

 __device__ __host__ void normalize(point &p)
    {
    float nor = norm(p);
    if ( nor == 0 ) return;
    p.x=p.x/nor;
    p.y=p.y/nor;
    p.z=p.z/nor;
    }

 __device__ __host__ point operator+(const point &left, const point &right)
    {
    point point;
    point.x = left.x + right.x;
    point.y = left.y + right.y;
    point.z = left.z + right.z;
    return point;
    }

 __device__ __host__ point operator-(const point &left, const point &right)
    {
    point point;
    point.x = left.x - right.x;
    point.y = left.y - right.y;
    point.z = left.z - right.z;
    return point;
    }

 __device__ __host__ point operator*(const float &scalar, const point &p)
    {
    point point;
    point.x = p.x * scalar;
    point.y = p.y * scalar;
    point.z = p.z * scalar;
    return point;
    }

 __device__ __host__ bool intersect(const ray &r, const point &punkt, const point &normale, point &intersection)
    {
    // sind sie (ann�hernd) parallel ?
    if ( dot(normale,r.richtung) < 0.00001 && dot(normale,r.richtung) > -0.00001 )
        return false;
    // Ebenengleichung aufstellen, Gerade einsetzen und somit l�nge der Richtung des
    // Richtungsvektor der Gerade errechnen
    float e = dot(normale,punkt-r.start)/dot(normale,r.richtung);
    // Schnittpunkt hinter Bildebene
    if (e < 0 ) return false;
    // Punkt errechnen 
    intersection = r.start + (e*r.richtung);
    return true;
    }

 __device__ __host__ bool intersect(const ray &r, const triangle &t, point &intersection)
    {
    if(!intersect(r,t.A,cross(t.A-t.B,t.A-t.C),intersection))
        {
        return false;
        }
    // die normale von A zu B und A zu C mal A zu P und A zu C ist positiv, wenn der Punkt innerhalb der gleichen Seite liegt
    // das ganze muss f�r alle drei seiten gelten
    if (      dot(cross(intersection-t.A,t.B-t.A),cross(t.C-t.A,t.B-t.A)) > 0
              && dot(cross(intersection-t.B,t.C-t.B),cross(t.A-t.B,t.C-t.B)) > 0
              && dot(cross(intersection-t.C,t.A-t.C),cross(t.B-t.C,t.A-t.C)) > 0 )
        return true;

    return false;
    }

 __device__ __host__ void initial_ray(const camera &c, int x, int y, ray &r,const point &stdRechts,const point &stdRunter)
    {
    // +0.5, damit der pixel in der mitte fixiert wird
    point schnittPunkt = c.obenLinks + (x+0.5)*stdRechts + (y+0.5)*stdRunter;
    // schnittpunkt mit der bildebene, damit objekte zwischen cam und bildebene nicht beachtet werden
    r.start=schnittPunkt;
    r.richtung = schnittPunkt - c.position;
    }

 __device__ __host__ void winkelRayTriangle(const ray &r, const triangle &t, float &winkel)
    {
    point p = cross(t.A-t.B,t.A-t.C);
    winkel = fabs(dot(r.richtung,p))/(norm(p)*norm(r.richtung));
    }

 __device__ __host__ rgb shade(const ray &r, const triangle &t)
    {
    float winkel;
    winkelRayTriangle(r,t,winkel);
    rgb shadeColor;
    shadeColor.r = winkel * t.farbe.r;
    shadeColor.g = winkel * t.farbe.g;
    shadeColor.b = winkel * t.farbe.b;
    return shadeColor;
    }
	
#if __CUDA__
 __host__ void copyImageToHost(rgb* image, rgb* imageDevPointer, int n)
    {
	int sizeInBytes = n*sizeof(rgb);
	cudaError_t error = cudaMemcpy(image, imageDevPointer, sizeInBytes, cudaMemcpyDeviceToHost);
	CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
	}
#endif	

#if __CUDA__
 __host__ void freeDeviceData(triangle* primsDevPointer, deviceData* devDatPointer, rgb* imageDevPointer)
    {
	cudaError_t error = cudaFree(primsDevPointer);
	CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
	error = cudaFree(devDatPointer);
	CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
	error = cudaFree(imageDevPointer);
	CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    LOG(INFO) << "Speicher auf dem Device freigegeben.";
	}
#endif	

 __host__ void erzeugeBildebene(scene &s, point &stdRechts, point &stdRunter, const int &height, const int &width)
	{
	point mittelPunkt;
    point obenMitte;
    point normUp = s.cam.oben;
    point normLeft;
	point normRichtung = s.cam.richtung;
	normalize(normRichtung);
    mittelPunkt=s.cam.position+s.cam.entfernung*normRichtung;
    normalize(normUp);
    float entfernungOben;
    float entfernungLinks;
    float PI = 3.141592653;
    entfernungOben = tan((s.cam.vertikalerWinkel/360)*PI)*s.cam.entfernung;
    entfernungLinks = tan((s.cam.horizontalerWinkel/360)*PI)*s.cam.entfernung;
    obenMitte = mittelPunkt + entfernungOben*normUp;
    normLeft=cross(s.cam.richtung,s.cam.oben);
    normalize(normLeft);
    s.cam.obenLinks = obenMitte + entfernungLinks*normLeft;
    // oben mitte - linksoben durch die h�lfte der weite
    stdRechts=(2.0/width)*(obenMitte-s.cam.obenLinks);
    // oben mitte - mitte durch die h�lfte der tiefe
    stdRunter=(2.0/height)*(mittelPunkt-obenMitte);
	}	
	
#if __CPU__
bool nextTriangle(const ray &r, const primitives &prims, const float &entfernung)
	{
	float EPSILON = 0.0001;
	point p;
    for (unsigned int o=0; o < prims.t.size(); o++ )
        {
        if ( intersect(r,prims.t[o],p) )
            {
            p = p - r.start;
            if ((norm(p)+EPSILON) < entfernung) // n�her dran als vorhergehendes objekt ?
                {
                return false;
                }
            }
        }
	return true;		
	}
#endif
	
#if __CUDA__
 __device__ bool nextTriangle(const ray r, const triangle* prims, const int primSize, const float entfernung)
	{
	// gibt es ein anderes Dreick zwischen dem Dreieck und dem Licht ?
	float EPSILON = 0.0001;
	point p;
    for (int o=0; o < primSize; o++ )
        {
        if ( intersect(r,prims[o],p) )
            {
            p = p - r.start;
            if ((norm(p)+EPSILON) < entfernung) // n�her dran als vorhergehendes objekt ?
                {
                return false;
                }
            }
        }
	return true;		
	}
#endif

#if __CUDA__
__global__ void renderImage(triangle* prims, deviceData* devDat, rgb* image, point* lights) 
	{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;  
	float winkel;
	float shadow;
	point p_save;
	triangle t;
	rgb farbe;
	if ( y < devDat->height && x < devDat->width )
		{
		t.A.x = INT_MAX; // Markierung f�r nicht gefundenes Dreieck
		shadow = 0;
		image[y*devDat->width+x]=devDat->hintergrund;
        ray r;
        initial_ray(devDat->cam,x,y,r,devDat->stdRechts,devDat->stdRunter);
        point p;
        float entfernung = FLT_MAX;
        for (int o=0; o < devDat->primSize; o++ )
            {
            if ( intersect(r,prims[o],p) )
                {
                if (norm(p - r.start) < entfernung) // n�her dran als vorhergehendes objekt ?
                    {
					p_save = p;
                    entfernung = norm(p - r.start);
                    t=prims[o];
                    }
                }
            }
		if ( t.A.x != INT_MAX )
			{
			farbe = t.farbe;//shade(r,t);
			for (int l=0; l < devDat->lightSize; l++ )
				{
				// Strahl vom Lichtpunkt zum Dreieck
				r.start = lights[l];
				r.richtung = p_save - r.start;	
				entfernung = norm(r.richtung);
				if ( nextTriangle(r, prims,devDat->primSize, entfernung) )
					{
					// Lichtintensit�t wird durch winkel Licht - Dreieck berrechnet
					// addiert sich bei mehreren Lichtquellen nat�rlich �ber 1 auf
					winkelRayTriangle(r,t,winkel);
					shadow = shadow + winkel;
					}
				if ( shadow > 1 ) break;
				}
			if ( shadow < 1 )	
				{
				// Reicht das licht nicht aus um die Farbe voll darzustellen wird sie abgeschw�cht
				farbe.r = shadow * farbe.r;
				farbe.g = shadow * farbe.g;
				farbe.b = shadow * farbe.b;
				}	
			image[y*devDat->width+x]=farbe;
			}
			
        }
	}
#endif

#if __CUDA__
 __host__ triangle* erzeugePrimitiveArray(primitives &prim)
	{
	int size = prim.t.size();
	triangle* t = (triangle*)malloc(size*sizeof(triangle));
	for (int i=0; i < size; i++)
		{
		t[i]=prim.t[i];
		}
	return t;
	}
#endif

#if __CUDA__
 __host__ point* erzeugeLightArray(lights &l)
	{
	int size = l.l.size();
	point* lights = (point*)malloc(size*sizeof(point));
	for (int i=0; i < size; i++)
		{
		lights[i]=l.l[i];
		}
	return lights;
	}
#endif

 __host__ void render_image(scene &s, const int &height, const int &width, rgb* image)
    {
    
	point stdRechts;
    point stdRunter;
	erzeugeBildebene(s, stdRechts, stdRunter, height, width);
	
	#if __CUDA__
	cudaGetLastError();
	cudaError_t error;
	// Daten die auf dem Device gebraucht werden erzeugen
	deviceData devDat;
	devDat.stdRechts = stdRechts;
	devDat.stdRunter = stdRunter;
	devDat.hintergrund = s.hintergrund;
	devDat.cam = s.cam;
	devDat.height = height;
	devDat.width = width;
	devDat.primSize = s.objekte.t.size();
	devDat.lightSize = s.lichter.l.size();
	triangle* t = erzeugePrimitiveArray(s.objekte);
	point* l = erzeugeLightArray(s.lichter);
	

    int sizeInBytes = s.objekte.t.size()*sizeof(triangle);
    triangle* primsDevPointer;
    error = cudaMalloc(&primsDevPointer, sizeInBytes);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaMemcpy(primsDevPointer, t, sizeInBytes, cudaMemcpyHostToDevice);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    CHECK_NOTNULL(primsDevPointer);
	
	sizeInBytes = s.lichter.l.size()*sizeof(point);
    point* lightDevPointer;
    error = cudaMalloc(&lightDevPointer, sizeInBytes);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaMemcpy(lightDevPointer, l, sizeInBytes, cudaMemcpyHostToDevice);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    CHECK_NOTNULL(lightDevPointer);

    sizeInBytes = sizeof(deviceData);
    deviceData* devDatPointer;
    error = cudaMalloc(&devDatPointer, sizeInBytes);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaMemcpy(devDatPointer, &devDat, sizeInBytes, cudaMemcpyHostToDevice);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
	CHECK_NOTNULL(devDatPointer);


	sizeInBytes = height*width*sizeof(rgb);
    rgb* imageDevPointer;
    error = cudaMalloc(&imageDevPointer, sizeInBytes);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
	CHECK_NOTNULL(imageDevPointer);
	LOG(INFO) << "Speicher auf dem Device angefordert.";
	int n=8;
	int x = ((height+n-1)/n);
	int y = ((width+n-1)/n);
	dim3 dimBlock(n,n);
	dim3 dimGrid(x,y);	

	LOG(INFO) << "Kernel gestartet.";
	renderImage<<<dimGrid, dimBlock>>>(primsDevPointer, devDatPointer, 
		imageDevPointer, lightDevPointer);	
	error = cudaGetLastError();
	CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
	error = cudaThreadSynchronize();
	CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
	
	copyImageToHost(image, imageDevPointer, width*height);
	LOG(INFO) << "Alle Threads abgearbeitet und Daten zur�ck kopiert.";

	// Speicher freigeben	
	freeDeviceData(primsDevPointer, devDatPointer, imageDevPointer);
	#endif
	
    #if __CPU__
	float winkel;
	ray r;
	rgb farbe;
	triangle t ;
	float shadow;
	float entfernung;
	for (int h=0; h < height; h++)
        {
        for (int w=0; w < width; w++)
            {
			t.A.x = INT_MAX; // Markierung f�r nicht gefundenes Dreieck
            image[h*width+w]=s.hintergrund;
			shadow = 0;
            initial_ray(s.cam,w,h,r,stdRechts,stdRunter);
            point p;
			point p_save;
            entfernung = FLT_MAX;
            for (unsigned int o=0; o < s.objekte.t.size(); o++ )
                {
                if ( intersect(r,s.objekte.t[o],p) )
                    {
                    if (norm(p - r.start) < entfernung) // n�her dran als vorhergehendes objekt ?
                        {
						p_save = p;
                        entfernung = norm(p - r.start);
                        t = s.objekte.t[o];
                        }
                    }
                }
			if ( t.A.x != INT_MAX )
				{
				farbe = t.farbe;//shade(r,t);
				for (unsigned int l=0; l < s.lichter.l.size(); l++ )
					{
					// ray vom lichtpunkt zum schnittpunkt mit dem Dreieck
					r.start = s.lichter.l[l];
					r.richtung = p_save - r.start;	
					entfernung = norm(r.richtung);
					if ( nextTriangle(r, s.objekte, entfernung) )
						{
						
						winkelRayTriangle(r,t,winkel);
						shadow = shadow + winkel;
						}
					if ( shadow > 1 ) break;
					}
				if ( shadow < 1 )	
					{
					farbe.r = shadow * farbe.r;
					farbe.g = shadow * farbe.g;
					farbe.b = shadow * farbe.b;
					}	
				image[h*width+w] = farbe;
				}
            }
        }
    LOG(INFO) << "Alle Pixel berechnet.";
	#endif
    }




