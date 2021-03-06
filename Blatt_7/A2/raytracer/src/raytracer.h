#pragma once

#include "types.h"

// vector operations
point cross(const point& p1, const point& p2);
float dot(const point& p1, const point& p2);
float norm(const point& p);
void normalize(point& p);
point operator+(const point& left, const point& right);
point operator-(const point& left, const point& right);
point operator*(const float& scalar, const point& p);

void angel(const ray& r, const triangle& t, float &angel);
rgb shade(const ray& r, const triangle& t);
// intersection
bool intersect(const ray &r, point vert0, point vert1, point vert2, point &inter);

// initial rays
void initial_ray(const camera& c, int x, int y, ray& r);

void render_image(scene& s, const int& height, const int& width, rgb* image);

