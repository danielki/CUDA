#pragma once

#include <yaml-cpp/yaml.h>

#include "types.h"

void operator >>(const YAML::Node& node, point& p);
void operator >>(const YAML::Node& node, rgb& r);
void operator >>(const YAML::Node& node, triangle& t);
void operator >>(const YAML::Node& node, camera& c);
void operator >>(const YAML::Node& node, ray& r);

void parse_scene(const char* filename, scene& s);
void find_primitives(const YAML::Node& doc, primitives& p);
void find_camera(const YAML::Node& doc, camera& c);
void find_background(const YAML::Node& doc, rgb& b);
