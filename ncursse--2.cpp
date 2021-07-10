#include <ncurses.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>

#include <chrono>
#include<cstdio>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

using namespace std;
//'#'/*



const char* texture_Hammer_Und_Sichel=
"...............WW..............."
"...............WW..............."
"...............WW..............."
"...............WW..............."
"...............WW..............."
"...............WW..............."
"WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"
"WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"
"WW.............................."
"WW.............................."
"WW.............................."
"WW.............................."
"WW.............................."
"WW.............................."
"WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"
"WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW";
const double EPS = 0.0001;

double scale = 100.;
double grav = -0.01;

double eq(double a, double b) {
	return (abs(abs(a)-abs(b)) <= EPS);
}

vector<vector<double>> buf;

void pix(int x, int y, int ch, int color_pair, double dist) {
	if (x >= COLS || x < 0 || y >= LINES || y < 0) {
		return;
	}
	if (buf[y][x] < dist) {
		return;
	} else {
		buf[y][x] = dist;

		attron(COLOR_PAIR(color_pair));
		mvaddch(LINES-y, x, ' ');
		attroff(COLOR_PAIR(color_pair));	
	}
}

void pix(int x, int y, int color_pair, double dist) {
	pix(x,y,'#',color_pair, dist);	
}

void pix(int x, int y, char Ch) {
	if (x >= COLS || x < 0 || y >= LINES || y < 0) {
		return;
	}
		mvaddch(y, x, Ch);
}

void pix(int x, int y) {
	pix(x,y,1,0.0)	;
}


struct Vec2 {
	double x;
	double y;
	Vec2(double x0, double y0): x(x0), y(y0) {}
};

struct Vec3 {
	double x;
	double y;
	double z;
	double l;
	Vec3(): x(0),y(0),z(0),l(0) {}
	Vec3(double x0, double y0, double z0): x(x0), y(y0), z(z0), l(sqrt(x*x + y*y + z*z)) {}
	Vec3(const Vec3& a): x(a.x), y(a.y), z(a.z), l(sqrt(x*x + y*y + z*z)) {}
	Vec3(const Vec2& v0): x(v0.x), y(v0.y), z(0), l(sqrt(x*x + y*y + z*z)) {}
	Vec3& operator+=(const Vec3& v) {
		x+=v.x;
		y+=v.y;
		z+=v.z;
		l = this -> length();
		return *this;
	}	
	Vec3& operator*=(const double a) {
		x*=a;
		y*=a;
		z*=a;
		l = this -> length();
		return *this;
	}	
	Vec3 operator-=(const Vec3& v) {
		return Vec3(x-=v.x,y-=v.y,z-=v.z);
	}	
	Vec3 operator/=(const double a) {
		return Vec3(x/=a,y/= a,z/=a );
	}	
	Vec3 operator/(const double& b) const{
		if (abs(b - 0) <= EPS) {
			double a = numeric_limits<double>::max();
			return Vec3(a,a,a);
		}
		return Vec3(x/b,y/b,z/b);
	}	
	friend Vec3 operator*(const double& a, const Vec3& v) {
		return Vec3(v.x*a, v.y*a, v.z*a);
	}
	friend Vec3 operator*( const Vec3& v, const double& a) {
		return Vec3(v.x*a, v.y*a, v.z*a);
	}
	friend Vec3 operator-(const Vec3& a, const Vec3& b) {
		return Vec3(a.x-b.x,a.y-b.y,a.z-b.z);
	}		
	friend Vec3 operator+(const Vec3& a, const Vec3& b) {
		return Vec3(a.x+b.x,a.y+b.y,a.z+b.z);
	}			
	double& operator[](int i) {
		if (i == 0) {
			return x;
		} else if (i == 1) {
			return y;
		} else return z;
	}
	const double& operator[](int i) const {
		if (i == 0) {
			return x;
		} else if (i == 1) {
			return y;
		} else return z;
	}
	double length() const{
		return sqrt(x*x + y*y + z*z);
	}
	Vec3 normalize() {
		double self_l = this->length();
		Vec3 self = *this;
		if (self_l == 0) return self;
		return self / self_l;
	}

	friend ostream& operator<<(ostream& o, const Vec3& v) {
		o << v.x << ' ' << v.y << ' ' << v.z << endl;
		return o;
	}
};

typedef int Color;

const Color BLACK = 0;

double dot(const Vec3& x, const Vec3& y) {
	return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
}

Vec3 cross(const Vec3& a, const Vec3& b) {
	return Vec3(
		a[1]*b[2] - a[2]*b[1],
	   -a[0]*b[2] + a[2]*b[0],
		a[0]*b[1] - a[1]*b[0]);
}

void line(int x0, int y0, int x1, int y1) {
	bool steep = false;
	if (abs(y1 - y0) > abs(x1 - x0)) {
		steep = true;
		swap(x0, y0);
		swap(x1, y1);
	}
             
	if (x1 < x0) {
		swap(x0, x1);
		swap(y0, y1);
	}

	double diff = 1. / (x1 - x0);
	int m10 = x1*y0;
	int m01 = x0*y1;

	for (int x = x0; x <= x1; x++) {
		int y = (int) (m10 - m01 - x*(y0 - y1)) * diff;
		if (steep) {
			pix(y,x); 
		} else {
			pix(x,y); 
		}
	}
}
struct Texture {
	vector<vector<int>> map;
	int height, width;
	Texture(int h, int w): height(h), width(w) {
		vector<vector<int>> m(height, vector<int>(width, '#'));
		map = m;
	}
	Texture(int h, int w, const char* c): height(h), width(w) {
		vector<vector<int>> m(height, vector<int>(width, '#'));
		map = m;
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				map[i][j] = c[w*i + j];
			}
		}
	}
	double get_color(const double a, const double b) const {
		int x = max(min(width*(0.5+b), width-1.),0.);
		int y = max(min((a+0.5)*height, height-1.),0.);
		if (y < height && x < width && x >= 0 && y >= 0) {
			if (map[y][x] == 'W') {
				return 1;
			} else if (map[y][x] == '.'){
				return 0.1;
			};
		}
		else {
			return '@'; //BLACK;
		}
	}
};

struct Plane { 
	Vec3 i, j, h, c;
	double width, height;
	Texture texture;
	Color defcol;
	Plane(const Vec3& c0, const Vec3& h0, const Texture& tx0, const Color def): 
	Plane(c0,h0,tx0) {
		defcol = def;
	}
	Plane(const Vec3& c0, const Vec3& h0, 
		const Texture& tx0): 
	Plane(c0,h0,1,1,tx0) {}
	Plane(const Vec3& c0, const Vec3& h0, double w0, double hei0,
		const Texture& tx0): 
	c(c0), h(h0), texture(tx0), defcol(1), width(w0), height(hei0 ) {
		i = cross(Vec3(0,1,0),h.normalize()).normalize();
		if (eq(i.l,0)) {
			i = Vec3(1,0,0);
			j = Vec3(0,0,1);
		} else {
			j = cross(h.normalize(),i.normalize()).normalize();
		}
		i *= width;
		j *= height;
	}
};

struct Matrix {
	vector<vector<double>> m;

	Matrix():
	m({{0,0,0}, {0,0,0}, {0,0,0}})
	{}

	Matrix(double a, double b, double c,
		   double d, double e, double f,
		   double g, double h, double i):
	m({{a,b,c}, {d,e,f}, {g,h,i}})
	{}

	Matrix(const Vec3& a0, const Vec3& b0, const Vec3& c0):
	m({{0,0,0}, {0,0,0}, {0,0,0}})
	{
		Vec3 a = a0;
		Vec3 b = b0;
		Vec3 c = c0;
		for (int i = 0; i < 3; i++) {
			m[i][0] = a[i];
			m[i][1] = b[i];
			m[i][2] = c[i];
		}
	}

	Vec3 operator*(const Vec3& v1) const {
		Vec3 v = v1;
		Vec3 res;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				res[i] += m[i][j]*v[j];

			}
		}					
		return res;
	}
	Matrix& operator*=(const Matrix& a) {
		Matrix res;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 3; k++) {
					res[i][j] += m[i][k]*a[k][j];
				}
			}
		}
		*this = res;
		return *this;
	}

	const Matrix& operator/=(double a) {
		if (a == 0) {
			a = EPS;
		}
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				m[i][j] /= a;
			}
		}
		return *this;
	}

	friend Matrix operator*(const Matrix& a, const Matrix& b) {
		Matrix m;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 3; k++) {
					m.m[i][j] += a.m[i][k]*b.m[k][j];
				}
			}
		}
		return m;
	}
		friend Matrix operator*(const double a, const Matrix& b) {
		Matrix m;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
					m.m[i][j] = a*b.m[i][j];
			}
		}
		return m;
	}
	double det() const {
		return m[0][0]*m[1][1]*m[2][2]
			+  m[0][1]*m[1][2]*m[0][2]
			+  m[1][0]*m[2][0]*m[2][1]
			-  m[0][2]*m[1][1]*m[2][0]
			-  m[0][0]*m[0][1]*m[1][0]
			 - m[2][2]*m[2][1]*m[1][2];
	}
	Matrix inverse() const {
		double invdet = this->det();
		Matrix minv;
		minv[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) / invdet;
		minv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / invdet;
		minv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / invdet;
		minv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / invdet;
		minv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / invdet;
		minv[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) / invdet;
		minv[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) / invdet;
		minv[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) / invdet;
		minv[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) / invdet;	
		return minv;
	}

	Matrix transpose() const{
		Matrix mt;
			for (int j = 0; j < 3; j++) {
				for (int i = 0; i < 3; i++){
					mt[i][j] = m[j][i];
				}
			}
		return mt;
	}

	vector<double>& operator[](int x) {
		return m[x];
	}
	vector<double> operator[](int x) const {
		return m[x];
	}
	friend ostream& operator<<(ostream& os, const Matrix& m) {
		os<<endl;
		for(int i=0;i<3;i++){
			for(int j=0;j<3;j++){
				os<<m[i][j]<<' ';
			}
			os<<endl;
		}os<<endl;
		return os;
	}
	double norm() const{
		double sum = 0;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				sum += m[i][j]*m[i][j];
			}
		}
		return sqrt(sum/3);		
	}
	const Matrix& normalize() {
		double norm = this->norm();
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					m[i][j] /= norm;
				}
			}			
		return *this;
	}
};

struct Cube {
	Plane top, bottom, left, right, hub, rim;
	vector<Plane> sides;
	Cube(const Vec3& c0, const Texture& tx):
		top(c0 + Vec3(0,-.5,0), Vec3(0,-1,0), tx, 20),
		bottom(c0 + Vec3(0,.5,0), Vec3(0,1,0), tx, 21),
		left(c0 + Vec3(-.5,0,0), Vec3(-1,0,0), tx, 22),
		right(c0 + Vec3(.5,0,0), Vec3(1,0,0), tx, 23),
		hub(c0 + Vec3(0,0,.5), Vec3(0,0,1), tx, 24),
		rim(c0 + Vec3(0,0,-.5), Vec3(0,0,-1), tx, 25),
		sides({top,bottom,right,left,hub,rim})
	{}
};

struct Light {
	Vec3 c;
	double intensity = 1.;
	Light(Vec3 c0, double in0):c(c0), intensity(in0) {}
};

typedef pair<double,double> dpair;

dpair make_line(double x0, double y0, double x1, double y1) {
	double diff = x1 - x0;
	if (eq(diff,0)) {
		diff = EPS;
	}

	if (diff < 0) {
		swap(x0,x1);
		swap(y0,y1);
	}
	diff = x1 - x0;
	double k = 1.*(y1 - y0)/diff;
	double b = 1.*(x1*y0 - x0*y1) / diff;
	return make_pair(k,b);
}

void draw_side(const Plane& plane, const Vec3& h, const Vec3& pos,
			const Matrix& M, const Matrix& M1, const Light& light,
			int bones=1) {

	Matrix transform=M1;
	Color color;
	Vec3 localC = plane.c - pos;
	/*if (localC.length() < 1){
		return ;
	}*/
	 /*
	if (dot(localC,h)/(localC.l*h.l)<0.2) {
		return ;
	}*/
	vector<Vec3> a1 = {localC-.5*plane.i-.5*plane.j,  
					   localC-.5*plane.i+.5*plane.j, 
					   localC+.5*plane.i-.5*plane.j, 
					   localC+.5*plane.i+.5*plane.j};

					   for (int i = 0; i < 4; i++) {
					   		if (dot(a1[i],h) < 0) {
					   			return;
					   			swap(a1[i].x,a1[i].y);
					   		}
					   }	

	double h2 = dot(h,h);		  
	for (int i = 0; i <= 3; i++) {
		a1[i] = M1*((h2 / dot(a1[i],h))*a1[i])*scale;
	}

	bool in_screen = false;/*
	for (int i = 0; i < 4; i++) {
		in_screen |= (a1[i].x+COLS/2 > -3*COLS || a1[i].x+COLS/2 < 4*COLS
		 || a1[i].y+LINES/2 > - 3*LINES || a1[i].y+LINES/2 <  4*LINES);
	}
	if (!in_screen) {
		return;
	}*/

	Vec3 c1((a1[0].x + a1[1].x + a1[2].x + a1[3].x)/4., 
		   (a1[0].y + a1[1].y + a1[2].y + a1[3].y)/4., 0);

	sort(a1.begin(), a1.end(), [c1](Vec3 a, Vec3 b) {
		return atan2(a.y-c1.y,a.x-c1.x) < atan2(b.y-c1.y,b.x-c1.x);
	});
	if (a1[0].x != min(a1[0].x,min(a1[1].x,min(a1[2].x,a1[3].x)))) {
		Vec3 ress = a1[0];
		for (int i = 0; i <= 2; i++) {
			a1[i] = a1[i+1];
		}
		a1[3] = ress;
	}	if (a1[0].x != min(a1[0].x,min(a1[1].x,min(a1[2].x,a1[3].x)))) {
		Vec3 ress = a1[0];
		for (int i = 0; i <= 2; i++) {
			a1[i] = a1[i+1];
		}
		a1[3] = ress;
	}	if (a1[0].x != min(a1[0].x,min(a1[1].x,min(a1[2].x,a1[3].x)))) {
		Vec3 ress = a1[0];
		for (int i = 0; i <= 2; i++) {
			a1[i] = a1[i+1];
		}
		a1[3] = ress;
	}
	dpair l1 = make_line(a1[0].x, a1[0].y,a1[1].x, a1[1].y);
	dpair l2 = make_line(a1[1].x, a1[1].y,a1[2].x, a1[2].y);
	dpair l3 = make_line(a1[2].x, a1[2].y,a1[3].x, a1[3].y);
	dpair l4 = make_line(a1[3].x, a1[3].y,a1[0].x, a1[0].y);
/*
	if (l4.first > 2) {
		l4 = l3;
	}
	if (l3.first > 2) {
		l3 = l4;
	}*/
	    auto t6 = high_resolution_clock::now();
	if (bones & 2) {
		line(a1[0].x+COLS/2,a1[0].y + LINES/2,a1[1].x+COLS/2,a1[1].y + LINES/2);
		line(a1[1].x+COLS/2,a1[1].y + LINES/2,a1[2].x+COLS/2,a1[2].y + LINES/2);
		line(a1[3].x+COLS/2,a1[3].y + LINES/2,a1[2].x+COLS/2,a1[2].y + LINES/2);
		line(a1[0].x+COLS/2,a1[0].y + LINES/2,a1[3].x+COLS/2,a1[3].y + LINES/2);
	}
/*
	double E_i, E_j, den, mul, a, b;
	double l_h = sqrt(h2);
	Vec3 e = h / l_h;	
	double dot_ce = dot(localC, e);
*/

	if (!(bones & 1)) {
		return;
	}

	Vec3 i = plane.i;
	Vec3 j = plane.j;
	double eps = 1. / h2;
	double dot_ch = dot(localC,h);
	double dot_cph = dot(localC, plane.h);
	/*if(dot_cph < 0) {
		dot_cph *= -1;
	}*/
	double dot_ci = dot(localC,i);
	double dot_cj = dot(localC,j);
	double a;
	double b;

	double epsdot_ch = eps*dot_ch;
	//Vec3 lightdir;

	double y; 
	Vec3 r;
	Vec3 xstep = M*Vec3(1.,0,0)/scale;
	Vec3 ystep = M*Vec3(0,1.,0)/scale;
	vector<dpair> l = {l1, l2, l3, l4};

	Vec3 lightdir = light.c - (plane.c);
double dot_l; /* = light.intensity*dot(lightdir, plane.h)
						/ (lightdir.l*lightdir.l*lightdir.l*plane.h.l + EPS);
*/
	for (int sk = 0; sk < 3; sk++) {
		if (a1[sk+1].x <= a1[sk].x)	break;
		auto t7 = high_resolution_clock::now();
		if (abs(l[sk].first) < 100) {
			for (int x = max(a1[sk].x,-1.*(COLS/2)-1) ; 
					 x <= min(a1[sk+1].x,1.*(COLS/2)+1); x++) 
			{
			 	y = l[sk].first*x + l[sk].second-1;	
				do {		
 				 	r = x*xstep + y*ystep + h; 
 				 	Vec3 locr = localC - r*(dot_cph/dot(r,plane.h));
					//E_i = dot(h,i) + dot(transform*r,i) / l_h;
					//E_j = dot(h,j) + dot(transform*r,j) / l_h;
					//den = 1 - dot(e,i)*E_i - dot(e,j)*E_j;
					//mul = dot_ce / den;
					////a = mul*E_i;
					////b = mul*E_j;
					 a = -dot(locr,i) ;/// i.l/i.l;
					 b = -dot(locr,j) ;/// j.l/i.l;
					// a = epsdot_ch*dot(r,i) - dot_ci;
					// b = epsdot_ch*dot(r,j) - dot_cj;
						//cerr << "a = " << a << ", b = " << b << endl;

						double tex_color = plane.texture.get_color(a,b);
					//Vec3 r1 = transform*r;	
					//pix(r1.x,r1.y,color,3);	
					//cerr << "x  "<<x<<", y "<<y << endl;
					//lightdir = light.c - (plane.c + a*i+b*j);
				dot_l = light.intensity*dot(lightdir, plane.h)
						/ (lightdir.l*lightdir.l*lightdir.l*plane.h.l + EPS);	
					if (dot_l < 0) {
						dot_l *= -1;
					} 

				double AMBIENT_LIGHT = 0.3;
					color = 24* max(2*atan(dot_l)/M_PI,0.3) * tex_color + 100;
					pix(x+COLS/2,y+LINES/2,color, (localC).l);
					y++;				    
				}	
				while (!( ( (sk <1) && (a1[2].x<a1[1].x) && !(y <= l[1].first*x +1+ l[1].second) ) || 
						(   (sk <2) &&(a1[3].x<a1[2].x) && !(y <= l[2].first*x +1+ l[2].second) ) ||
						( (1) && !(y <= l[3].first*x +1+ l[3].second) ) ));	
				
			} 
		}
	}
}

void draw_cube(const Cube& cube, const Vec3& h, const Vec3 pos,
			const Matrix& transform, const Matrix& t_i, const Light& light) {
	draw_side(cube.top,h,pos,transform, t_i, light);
	draw_side(cube.bottom,h,pos,transform, t_i, light);
	draw_side(cube.left,h,pos,transform, t_i, light);
	 draw_side(cube.right,h,pos,transform, t_i, light);
	 draw_side(cube.rim,h,pos,transform, t_i, light);
	 draw_side(cube.hub,h,pos,transform, t_i, light);
}

struct Map {
	vector<vector<vector<Cube*>>> map;
	int x0;
	int y0;
	int z0;
	Map(int x, int y, int z): x0(x-1),y0(y-1),z0(z-1)
	{
		map = vector<vector<vector<Cube*>>> (x,
			vector<vector<Cube*>> (y,
				vector<Cube*> (z)));
	}

	Cube* operator() (int x, int y, int z) {
		if (x > x0 || x < 0 ||
			y > y0 || y < 0 || 
			z > z0 || z < 0) {
			return NULL;
		} else 
			return map[x][y][z];
	}
	void add(int x, int y, int z, Texture tex) {
		if (x > x0 || x < 0 ||
			y > y0 || y < 0 || 
			z > z0 || z < 0) {
			return;
		}
		map[x][y][z] = new Cube(Vec3(x,y,z), tex);
	}
};

Matrix rotxz(double phi) {
	return Matrix(cos(phi),0,-sin(phi), 
				 0,       1,0        , 
				 sin(phi),0,cos(phi));
}

Matrix rotyz(double phi) {
	return Matrix(1,		0,		0, 
				  0,cos(phi),-sin(phi), 
				  0,sin(phi),cos(phi));
}

int main(int argc, char* argv[]) {

	initscr();
	start_color();

	timeout(50);

	int screen_width,screen_height;
	getmaxyx(stdscr,screen_height,screen_width);

	raw();
	keypad(stdscr, TRUE);
	noecho();
for (int i = 0; i <= 23; i++) {
	init_pair(100+i, COLOR_BLACK,232+i);
	attron(COLOR_PAIR(100+i));
	mvaddch(i,0,'@');
}
init_pair(124, COLOR_BLACK,16);
	attron(COLOR_PAIR(124));
	mvaddch(24,0,'@');

	init_pair(1, COLOR_RED, COLOR_RED);
	init_pair(2, COLOR_GREEN, COLOR_BLUE);
	init_pair(3, 16, 16);	

	init_pair(20, COLOR_BLACK, COLOR_BLACK);
	init_pair(21, COLOR_RED, COLOR_RED);
	init_pair(22, COLOR_WHITE, COLOR_WHITE);
	init_pair(23, COLOR_GREEN, COLOR_GREEN);
	init_pair(24, COLOR_BLUE, COLOR_BLUE);
	init_pair(25, COLOR_MAGENTA, COLOR_MAGENTA);		



	Texture texture(16,32,texture_Hammer_Und_Sichel);
	Vec3 pos(-0,10,0);	





	int success = 0;
	Plane near_plane(Vec3(-5,2,5), Vec3(0, -1,0), 5, 5,texture);
	Map map(12,12,12);
		for (int x = 0; x < 12; x++) {
			for (int z = 0; z < 12; z++) {
				map.add(x,0,z,texture);
			}
		}

		Cube box1(Vec3(0,0,5), texture);
		Cube box2(Vec3(1,1,5), texture);
		Plane ground(Vec3(-5,2,5), Vec3(0, -1,0), 5, 5,texture);

		vector<Cube> Box;
		for (int i = 1; i < 2; i++) {
			for (int j = 5; j < 6; j++) {
				Box.push_back(Cube(Vec3(j,i,2),texture));
			}
		}
		Vec3 h(1,0,0);
		Matrix transform(0,0,-1 , 0,1,0 , 1,0,0);

		Vec3 C = cross(Vec3(0,1,0),h.normalize()).normalize();
		Vec3 D = cross(h.normalize(),C.normalize()).normalize();
		transform = Matrix(-1*C.normalize(),D.normalize(),h.normalize());
		Matrix transform_inverse = transform.inverse();

		Matrix t1 = transform;
		double phi = M_PI / 100;
		Matrix r(cos(phi),0,-sin(phi), 
				 0,       1,0        , 
				 sin(phi),0,cos(phi));
		Matrix rb(cos(-phi),0,-sin(-phi), 
				  0,		1,0			, 
				  sin(-phi),0,cos(-phi));
		Matrix v0(1,0,0, 0,cos(phi),-sin(phi), 0,sin(phi),cos(phi));
		Matrix b0(1,0,0, 0, cos(phi),sin(phi), 0,-sin(phi),cos(phi));

		Light light(Vec3(1.6,-0.5,5), 100);
		//Cube box3(Vec3(0,-2,3), texture, 0.1);

		int x0 = 0;
		int y0 = 0;

	double t = 0;
	if (t == 3600) {
		t = 0;
	}
	
	Vec3 vsped;

	int ch;
	do {
		t += 0.2;

		pos += vsped;
		if(1) {/*auto i = ground;
			if (abs(pos.x-i.c.x)<i.width  &&
				abs(pos.z-i.c.z)<i.height &&
				abs(-pos.y+i.c.y)>EPS && -pos.y+i.c.y<1.) 
			{
				vspeed = 0;
			}*/

			if (map(pos.x,pos.y-1,pos.z) != NULL) {
				vsped.y = 0 ;
			vsped.x *= 0.5;
			vsped.z *= 0.5;	
			} else {
				vsped.y += grav;	
			}

		}



		light.c = pos + h;

		double dth = M_PI / 360.;
		double dphi = M_PI / 360.;
		double phi = atan2(h.x,h.z);
		double theta = atan2(sqrt(h.x*h.x+h.z*h.z),h.y);
		double rad = sqrt(h.x*h.x+h.z*h.z);


		ch = getch();
		Matrix v = transform*v0*transform_inverse;
		Matrix b = transform*b0*transform_inverse;

		Matrix v1 = t1*v0*t1.inverse();
		Matrix b1 = t1*b0*t1.inverse();

		buf = vector<vector<double>> (screen_height+1, 
				vector<double> (screen_width+1,numeric_limits<double>::max()));
		clear();	
			wbkgd(stdscr,COLOR_PAIR(2));	
		if (ch == ' ') {
			vsped.y = 0.15;
		}
		if (ch == '2') {
			
			t1 *= v1;
			 if (theta < M_PI*0.9) {
				h = v*h;
			}
		} 
		if (ch == '8') {
			//transform = b*transform;
			//transform_inverse = transform.inverse() ;

			t1 *= b1;
			if (theta > M_PI*0.1 + EPS) {
			/*h.y = cos(theta + dth);
				h.x = sin(theta + dth)*sin(phi);
				h.z = sin(theta + dth)*cos(phi);*/
							h = b*h;
			} 
		}
		if (ch == '6') {
			//transform = r*transform;
			//transform_inverse = transform.inverse() ;
			/*h.x = sin(phi - dphi);
			h.z = cos(phi - dphi);*/
			h = r*h;
			t1 *= r;
		}
		if (ch == '4') {
			//transform = rb*transform;
			//transform_inverse = transform.inverse() ;
			/*h.x = sin(phi + dphi);
			h.z = cos(phi + dphi);*/
			h = rb*h;
			t1 *= rb;
		}
		if (ch == '1') {
			//transform = b*r*transform;
			//transform_inverse = transform.inverse();
			//h = b*r*h;
		}
		if (ch == '7') {
			//transform = v*r*transform;
			//transform_inverse = transform.inverse();
			//h = v*r*h;
		}
		if (ch == '9') {
			//transform = v*rb*transform;
			//transform_inverse = transform.inverse();
			//h = v*rb*h;
		}
		if (ch == '3') {
			//transform = b*rb*transform;
			//transform_inverse = transform.inverse();
			//h = b*rb*h;
		}
		if(ch == 's' || ch == 'j') {
			vsped -= 0.1*Vec3(h.x,0,h.z).normalize();
		}
		if(ch == 'w' || ch == 'k') {
			vsped += 0.1*Vec3(h.x,0,h.z).normalize();
		}

		if (ch == 'a') {
			scale += 0.5;
		}
		if (ch == 'd') {
			scale += -0.5;
		}
		if (ch == 'z') {
			h *= 1.1;
		}
		if (ch == 'x') {
			h /= 1.1;
		}

		if (ch == 'f'/* && success*/) {
			//Box.push_back(Cube(near_plane.c + near_plane.h*.5, texture));
			Vec3 r = near_plane.c + near_plane.h*.5;
			map.add(r.x,r.y,r.z, texture);
		}

		Vec3 C = cross(Vec3(0,1,0),h).normalize();
		/*double bx = h.y*h.x / sqrt((h.x*h.x + h.z*h.z)*(1+h.y*h.y) + 2*h.x*h.z);
		double bz = h.y*h.z / sqrt((h.x*h.x + h.z*h.z)*(1+h.y*h.y) + 2*h.x*h.z);
		Vec3 d = {bx,
		 		(h.x + h.z)*bx / (h.y * h.x),
		 		bz};*/
		Vec3 D = cross(h,C).normalize();
		transform = Matrix(-1*C,D,h.normalize());

		transform_inverse = transform.transpose();

		draw_cube(box1,h,pos,transform,transform_inverse, light);
		draw_cube(box2,h,pos,transform,transform_inverse,light);

		for (auto b: Box) {
			draw_cube(b,h,pos,transform,transform_inverse,light);
		}

		for (int x = 0; x < 12; x++) {
			for (int y = 0; y < 12; y++) {
				for (int z = 0; z < 12; z++) {
					if (NULL != map(x,y,z))
					draw_cube(*map(x,y,z),h,pos,transform,transform_inverse,light);
				}
			}
		}

		double mindist = numeric_limits<double>::max();
		near_plane = (*Box.begin()).sides[0];
		success = 0;
		for (int x = 0; x < 12; x++) {
		for (int y = 0; y < 12; y++) {
		for (int z = 0; z < 12; z++) {
		if (NULL != map(x,y,z))    
		{
			for (int i = 0; i < 6; i++) {
				Plane b = map(x,y,z) -> sides[i];
				Vec3 Hnew = b.h*dot(b.h, b.c-pos);
				Matrix M1 = Matrix(b.h,b.i.normalize(),b.j.normalize()).transpose();
				Vec3 ray_cast = h*dot(Hnew, Hnew)/dot(h,Hnew);
				if (dot(ray_cast, h) <= 0) {
					continue;
				}
				Vec3 ray_trace = M1*(ray_cast - b.c + pos);
				if (ray_trace.y < b.width/2. && ray_trace.y > -b.width/2. &&
					ray_trace.z < b.height/2. && ray_trace.z> -b.height/2.) {
					if (ray_cast.l < mindist) {
						mindist = ray_cast.l;
						near_plane = b;
						success = 1;
					}
				}
			}
		}}}}
		if (success)
		draw_side(near_plane,h,pos,transform,transform_inverse,light,2);


		Plane d1(Vec3(-5,-1,5), Vec3(sin(t/20.), cos(t/20.),sin(t/10.)+2*cos(t/10.3)), 3, 3,texture);
		draw_side(d1,h,pos,transform,transform_inverse,light);

		draw_side(ground,h,pos,transform,transform_inverse,light);
		//mvprintw(3,0,"theta = %f\nphi = %f", theta, phi);
		mvprintw(3,0,"scale= %f", scale);
		mvprintw(5,0,"      C       D       h        pos");
		mvprintw(6,0,"x %5f %5f %5f %4f", C.x, D.x, h.x, pos.x);
		mvprintw(7,0,"y %5f %5f %5f %5f", C.y, D.y, h.y, pos.y);
		mvprintw(8,0,"z %5f %5f %5f %5f", C.z, D.z, h.z, pos.z );
		refresh();
	} while('q' != ch);

/*
	for (int i = 0; i < 255; i++) {
		for (int j = 0; j < 255; j++) {
			init_pair(i+256*j, j,i);
			attron(COLOR_PAIR(i+256*j));
			mvaddch(i,j,' ');
		}
	}
*/
	refresh();

	//getch();
	endwin();

	return 0;
}

