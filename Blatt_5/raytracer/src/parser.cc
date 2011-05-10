#include <glog/logging.h>
#include <fstream>
#include "parser.h"

void operator >>(const YAML::Node& node, point& v)
    {
    node[0] >> v.x;
    node[1] >> v.y;
    node[2] >> v.z;
    }

void operator >>(const YAML::Node& node, rgb& r)
    {
    node[0] >> r.r;
    node[1] >> r.g;
    node[2] >> r.b;
    }

void operator >>(const YAML::Node& node, triangle& t)
    {
    node[0] >> t.A;
    node[1] >> t.B;
    node[2] >> t.C;

    }

void operator >>(const YAML::Node& node, primitives& p)
    {
    triangle t;
    node["triangle"] >> t;
    node["color"] >> t.farbe;
    p.t.push_back(t);

    }

void operator >>(const YAML::Node& node, camera& c)
    {
    node["location"] >> c.position;
    node["direction"] >> c.richtung;
    node["up"] >> c.oben;
    node["distance"] >> c.entfernung;
    node["horizontal_angle"] >> c.horizontalerWinkel;
    node["vertical_angle"] >> c.vertikalerWinkel;
    }

void parse_scene(const char* filename, scene& s)
    {
    std::ifstream scene(filename);
    YAML::Parser parser(scene);
    YAML::Node doc;
    parser.GetNextDocument(doc);
    find_background(doc,s.hintergrund);
    find_camera(doc,s.cam);
    find_primitives(doc,s.objekte);
    }

void find_primitives(const YAML::Node& doc, primitives& p)
    {
    if (const YAML::Node& prims = doc["primitives"])
        {
        for(unsigned i=0; i<prims.size(); i++)
            {
            prims[i] >> p;
            }
        }
    }

void find_camera(const YAML::Node& doc, camera& c)
    {
    doc["camera"] >> c;
    }

void find_background(const YAML::Node& doc, rgb& b)
    {
    doc["background"] >> b;
    }
