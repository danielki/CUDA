#include <glog/logging.h>
#include <fstream>
#include "parser.h"
#include <stdio.h>
#include <stdlib.h>

void operator >>(const YAML::Node& node, point& v)
    {
    node[0] >> v.x;
    node[1] >> v.y;
    node[2] >> v.z;
    }

void operator >>(const YAML::Node& node, rgb& c)
    {
    int r,g,b;
    node[0] >> r;
    node[1] >> g;
    node[2] >> b;
    c.x=r;
    c.y=g;
    c.z=b;
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
    const YAML::Node *distance = node.FindValue("distance");
    if ( distance != NULL )
        {
        (*distance) >> c.entfernung;
        }
    else
        {    
        LOG(INFO) << "Kein Distance definiert, Standartwert 1 definiert.";
        c.entfernung = 1.0;
        }
    const YAML::Node *horizontal_angle = node.FindValue("horizontal_angle");
    if ( distance != NULL )
        {
        (*horizontal_angle) >> c.horizontalerWinkel;
        }
    else
        {   
        c.horizontalerWinkel = 140.0; 
        LOG(INFO) << "Kein horizontaler Winkel definiert, Standartwert 140 festgelegt.";
        }
    const YAML::Node *vertical_angle = node.FindValue("vertical_angle");
    if ( distance != NULL )
        {
        (*vertical_angle) >> c.vertikalerWinkel;
        }
    else
        {    
        c.vertikalerWinkel = 140.0;
        LOG(INFO) << "Kein vertikaler Winkel definiert, Standartwert 140 festgelegt.";
        }
    }

void parse_scene(const char* filename, scene& s)
    {
    std::ifstream scene(filename);
	if ( scene.is_open() )
		{
		YAML::Parser parser(scene);
		YAML::Node doc;
		parser.GetNextDocument(doc);
		find_background(doc,s.hintergrund);
		find_camera(doc,s.cam);
		find_lights(doc,s.lichter);
		find_primitives(doc,s.objekte);
		}
	else
		{
		LOG(INFO) << "Datei nicht gefunden.";
		exit (1);
		}
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
    else
        {    
        LOG(INFO) << "Keine Objekte definiert.";
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
    else
        {    
        LOG(INFO) << "Keine Lichter definiert.";
        }
    }

void find_camera(const YAML::Node& doc, camera& c)
    {
    const YAML::Node *cam = doc.FindValue("camera");
    if ( cam != NULL )
        {
        (*cam) >> c;
        }
    else
        {    
        LOG(INFO) << "Keine Kamera definiert.";
        }
    }

void find_background(const YAML::Node& doc, rgb& b)
    {
    const YAML::Node *background = doc.FindValue("background");
    if ( background != NULL )
        {
        (*background) >> b;
        }
    else
        {    
        LOG(INFO) << "Kein Hintergrund definiert.";
        }
    }
