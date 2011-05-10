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
    
void operator >>(const YAML::Node& node, lights& l)
    {
    point p;
    node >> p;
    l.l.push_back(p);
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
    find_lights(doc,s.lichter);
    find_primitives(doc,s.objekte);
    }

void find_primitives(const YAML::Node& doc, primitives& p)
    {
    const YAML::Node *prims = doc.FindValue("primitives");
    if ( prims != NULL )
        {
        for(unsigned i=0; i<(*prims).size(); i++)
            {
            (*prims)[i] >> p;
            }
        }
    }

void find_lights(const YAML::Node& doc, lights& l)
    {
    const YAML::Node *lights = doc.FindValue("lights");
    if ( lights != NULL )
        {
        for(unsigned i=0; i<(*lights).size(); i++)
            {
            (*lights)[i] >> l;
            }
        }
    }

void find_camera(const YAML::Node& doc, camera& c)
    {
    const YAML::Node *cam = doc.FindValue("camera");
    if ( cam != NULL )
        {
        (*cam) >> c;
        }
    }

void find_background(const YAML::Node& doc, rgb& b)
    {
    const YAML::Node *background = doc.FindValue("background");
    if ( background != NULL )
        {
        (*background) >> b;
        }
    }
