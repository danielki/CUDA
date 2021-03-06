#pragma once

#include <vector>
#include <iostream>
#include "vector_types.h"

// define structs
typedef float3 point;

struct rgb
    {
    int r;
    int g;
    int b;
    };

struct triangle
    {
    point A;
    point B;
    point C;
    rgb farbe;
    };

struct primitives
    {
    std::vector <triangle> t;
    };

struct camera
    {
    point position;
    point richtung;
    point oben;
    point obenLinks;
    float entfernung;
    float horizontalerWinkel;
    float vertikalerWinkel;
    };

struct ray
    {
    point start;
    point richtung;
    };

struct scene
    {
    rgb hintergrund;
    camera cam;
    primitives objekte;
    };

// define types
typedef struct rgb rgb;
typedef struct triangle triangle;
typedef struct primitives primitives;
typedef struct camera camera;
typedef struct ray ray;
typedef struct scene scene;

// define stream output for types
std::ostream& operator <<(std::ostream& s, const point& p);
std::ostream& operator <<(std::ostream& s, const rgb& r);
std::ostream& operator <<(std::ostream& s, const triangle& t);
std::ostream& operator <<(std::ostream& s, const camera& c);
std::ostream& operator <<(std::ostream& s, const ray& r);

