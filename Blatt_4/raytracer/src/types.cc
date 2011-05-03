#include "types.h"

std::ostream& operator <<(std::ostream& s, const point& p) {
	s << p.x << "," << p.y << "," << p.z;
	return s;
}

std::ostream& operator <<(std::ostream& s, const rgb& r) {
	s << r.r << " "<< r.g <<" " << r.b;
	return s;
}

std::ostream& operator <<(std::ostream& s, const triangle& t) {
	s << "A=" << t.A << ", B=" << t.B << ", C=" << t.C <<", Farbe=" << t.farbe;
	return s;
}

std::ostream& operator <<(std::ostream& s, const camera& c) {
    s << "Position=" << c.position << ", Richtung=" << c.richtung
      << ", Oben=" << c.oben << ", Obenlinks=" << c.obenLinks;
	return s;
}

std::ostream& operator <<(std::ostream& s, const ray& r) {
    s << "Start=" << r.start << ", Richtung=" << r.richtung;
	return s;
}

