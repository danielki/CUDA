#include <glog/logging.h>
#include <cmath>

#include "raytracer.h"

//WE ASSUME LEFT-HANDED ORIENTATION (left hand rule)...
point cross(const point& p1, const point& p2)
    {
    point point;
    point.x = p1.y*p2.z - p1.z*p2.y;
    point.y = p1.z*p2.x - p1.x*p2.z;
    point.z = p1.x*p2.y - p1.y*p2.x;
    return point;
    }

float dot(const point& p1, const point& p2)
    {
    float f;
    f = p1.x*p2.x+p1.y*p2.y+p1.z*p2.z;
    return f;
    }

float norm(const point& p)
    {
    float f;
    f = sqrt((p.x*p.x)+(p.y*p.y)+(p.z*p.z));
    return f;
    }

void normalize(point& p)
    {
    float nor = norm(p);
    if ( nor == 0 ) return;
    p.x=p.x/nor;
    p.y=p.y/nor;
    p.z=p.z/nor;
    }

point operator+(const point& left, const point& right)
    {
    point point;
    point.x = left.x + right.x;
    point.y = left.y + right.y;
    point.z = left.z + right.z;
    return point;
    }

point operator-(const point& left, const point& right)
    {
    point point;
    point.x = left.x - right.x;
    point.y = left.y - right.y;
    point.z = left.z - right.z;
    return point;
    }

point operator*(const float& scalar, const point& p)
    {
    point point;
    point.x = p.x * scalar;
    point.y = p.y * scalar;
    point.z = p.z * scalar;
    return point;
    }

bool pointSameSide(point A, point v2, point p)
    {

    return false;
    }

bool intersect(const ray& r, const point& punkt, const point& normale, point& intersection)
    {
    // sind sie (annähernd) parallel ?
    if ( dot(normale,r.richtung) < 0.00001 && dot(normale,r.richtung) > -0.00001 )
        return false;
    // Ebenengleichung aufstellen, Gerade einsetzen und somit länge der Richtung des
    // Richtungsvektor der Gerade errechnen
    float e = dot(normale,punkt-r.start)/dot(normale,r.richtung);
    if (e < 0 ) return false;
    // Punkt errechnen mit der länger des Vektors
    intersection = r.start + (e*r.richtung);
    return true;
    }

bool intersect(const ray& r, const triangle& t, point& intersection)
    {
    if(!intersect(r,t.A,cross(t.A-t.B,t.A-t.C),intersection))
        {
        return false;
        }
    // die normale von A zu B und A zu C mal A zu P und A zu C ist positiv, wenn der Punkt innerhalb der gleichen Seite liegt
    // das ganze muss für alle drei seiten gelten
    if (      dot(cross(intersection-t.A,t.B-t.A),cross(t.C-t.A,t.B-t.A)) > 0
              && dot(cross(intersection-t.B,t.C-t.B),cross(t.A-t.B,t.C-t.B)) > 0
              && dot(cross(intersection-t.C,t.A-t.C),cross(t.B-t.C,t.A-t.C)) > 0 )
        return true;

    return false;
    }

void initial_ray(const camera& c, int x, int y, ray& r,const point& stdRechts,const point& stdRunter)
    {
    // +0.5, damit der pixel in der mitte fixiert wird
    point schnittPunkt = c.obenLinks + (x+0.5)*stdRechts + (y+0.5)*stdRunter;
    // schnittpunkt mit der bildebene, damit objekte zwischen cam und bildebene nicht beachtet werden
    r.start=schnittPunkt;
    r.richtung = schnittPunkt - c.position;
    }

void render_image(scene& s, const int& height, const int& width, rgb* image)
    {

    point mittelPunkt;
    point obenMitte;
    point normUp = s.cam.oben;
    point normLeft;
    mittelPunkt=s.cam.position+s.cam.entfernung*s.cam.richtung;
    normalize(normUp);
    float entfernungOben;
    float entfernungLinks;
    entfernungOben = tan(s.cam.horizontalerWinkel/2)*norm(s.cam.entfernung*s.cam.richtung);
    entfernungLinks = tan(s.cam.vertikalerWinkel/2)*norm(s.cam.entfernung*s.cam.richtung);
    obenMitte = mittelPunkt + entfernungOben*normUp;
    normLeft=cross(s.cam.richtung,s.cam.oben);
    normalize(normLeft);
    s.cam.obenLinks = obenMitte + entfernungLinks*normLeft;
    //  oben mitte - linksoben durch die hälfte der weite
    point stdRechts=(2.0/width)*(obenMitte-s.cam.obenLinks);
    // oben mitte - mitte durch die hälfte der tiefe
    point stdRunter=(2.0/height)*(mittelPunkt-obenMitte);

    for (int h=0; h < height; h++)
        {
        for (int w=0; w < width; w++)
            {
            image[h*width+w]=s.hintergrund;
            ray r;
            initial_ray(s.cam,w,h,r,stdRechts,stdRunter);
            point p;
            float entfernung = 0;
            for (unsigned int o=0; o < s.objekte.t.size(); o++ )
                {
                if ( intersect(r,s.objekte.t[o],p) )
                    {
                    p = p - s.cam.position;
                    // Gleich 0 geht bei float und 0 iwie nicht ... equals 0 und epsilon ?
                    if (entfernung < 0.0000001 || norm(p) < entfernung) // näher dran als vorhergehendes objekt ?
                        {
                        entfernung = norm(p);
                        image[h*width+w]=s.objekte.t[o].farbe;
                        }
                    }
                }
            }
        }
    }




