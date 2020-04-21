//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
//
//
//Used material:
//My own engines base object for runtime type checks:
//		https://github.com/dpeter99/ShadowEngine/blob/master/ShadowEngine/src/Core/SHObject.h
//
//
#include "framework.h"

#ifdef LOCAL

#include <iostream>
#include <fstream>

#endif // LOCAL



#define RAY_T_MIN 0.0001f

#define RAY_T_MAX 1.0e30f

#define TRACE_MAX_ITERATION 10

#define OUT

#define CONSOLE_RENDER false

inline float pow2(float n)
{
	return n * n;
}

inline mat4 dot(const vec3 & P, const mat4& A)
{
	return mat4(
		vec4(A.rows[0].x * P.x * P.x,	A.rows[0].y * P.x * P.y,	A.rows[0].z * P.x * P.z,	A.rows[0].w * P.x),
		vec4(A.rows[1].x * P.y * P.x,	A.rows[1].y * P.y * P.y,	A.rows[1].z * P.y * P.z,	A.rows[1].w * P.y),
		vec4(A.rows[2].x * P.z * P.x,	A.rows[2].y * P.z * P.y,	A.rows[2].z * P.z * P.z,	A.rows[2].w * P.z),
		vec4(A.rows[3].x * P.x,		A.rows[3].y * P.y,		A.rows[3].z * P.z,		A.rows[3].w * 1.0f)
	);
}


template<class T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi)
{
	//assert(!(hi < lo));
	return (v < lo) ? lo : (hi < v) ? hi : v;
}

vec3 reflect(vec3 dir, const vec3& normal)
{
	vec3 I = dir;
	I = normalize(dir);
	return I - 2 * normal * (dot(I, normal));
}


#pragma region vec2 Ext

inline vec2 operator*=(const vec2& v, float a) {
	return vec2(v.x * a, v.y * a);
}

inline vec2 operator+=(const vec2& v, float a) {
	return vec2(v.x + a, v.y + a);
}

inline vec2 operator+=(const vec2& v, vec2 a) {
	return vec2(v.x + a.x, v.y + a.y);
}

inline bool operator==(const vec2& v, vec2 a) {
	return v.x == a.x && v.y == a.y;
}

#pragma endregion

#pragma region vec3 Ext

inline float length2(const vec3& v) { return dot(v, v); }

//Double sided operators that are missing ( +, -)
inline vec3 operator+(const vec3& v, float a) {
	return vec3(v.x + a, v.y + a, v.z + a);
}

inline vec3 operator-(const vec3& v, float a) {
	return vec3(v.x - a, v.y - a, v.z - a);
}


//The equals operators for floats
inline vec3 operator+=(const vec3& v, float a) {
	return v + a;
}

inline vec3 operator-=(const vec3& v, float a) {
	return v - a;
}

inline vec3 operator*=(const vec3& v, float a) {
	return v * a;
}

inline vec3 operator/=(const vec3& v, float a) {
	return v / a;
}

bool operator==(const vec3& a, const vec3& b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z;
}

//The equals operators for vec3s
//TODO: NOT working
inline vec3 operator-=(const vec3& v, const vec3& a) {
	return vec3(v.x - a.x, v.y - a.y, v.z - a.z);;
}

inline float normalize_this(vec3& v)
{
	float l = length(v);
	v /= l;
	return l;
}

inline float distance(vec3 point1, vec3 point2)
{
	float xD = point1.x - point2.x;
	float yD = point1.y - point2.y;
	float zD = point1.z - point2.z;
	float result = sqrt((xD * xD) + (yD * yD) + (zD * zD));
	return result;
}

#pragma endregion

#pragma region BaseObject

/**
 * \brief This is the base class for every class in the Engine that uses runtime reflection.
 * Currently it provides a runtime TypeID and TypeName witch can be accesed as static and as class memebers.
 * The ID is a int type number witch is generated incramently, on the first call to get a type.
 * Each class that inherits from this or it's parent inheris form it must implement the
	SHObject::GetType and SHObject::GetTypeId methodes and make it's own static methodes.
	To make it easier a standard implementation of these can be used with the SHObject_Base() macro
	witch implements all of these functions. It uses the typeid().name of the class.
 */
class SHObject
{
protected:
	/**
	 * \brief Generates a new UID for each call
	 * \return the next Unique ID that was just generated
	 */
	static uint64_t GenerateId() noexcept
	{
		static uint64_t count = 0;
		return ++count;
	}

public:
	/**
	 * \brief Returns the top level class type name of the object
	 * \return The class Class name as a string
	 */
	virtual const std::string& GetType() const = 0;
	/**
	 * \brief Gets the top level type ID
	 * \return UID of the class
	 */
	virtual const uint64_t GetTypeId() const = 0;

	virtual ~SHObject() = default;
};


/**
 * \brief Macro to make the override functions of SHObject. This should be added in each derived class
 * \param type The type of the class
 */
#define SHObject_Base(type)	\
public: \
	static const std::string& Type()				{ static const std::string t = #type; return t; } \
	static uint64_t TypeId()						{ static const uint64_t id = GenerateId(); return id; } \
	const std::string& GetType() const override		{ return Type();  } \
	const uint64_t GetTypeId() const override		{ return  type::TypeId(); } \
private:

#pragma endregion

class CoordinateSystem
{
public:
	static const vec3 Up;

	static const vec3 Forward;

	static const vec3 Right;

};
const vec3 CoordinateSystem::Up = vec3(0, 1, 0);
const vec3 CoordinateSystem::Forward = vec3(0, 0, 1);
const vec3 CoordinateSystem::Right = vec3(1, 0, 0);


/// <summary>
/// Represents a ray that originates from a point in space and travels in a direction 
/// </summary>
class Ray
{
public:

	vec3 origin;
	vec3 direction;
	float tMax;

	Ray() :
		origin(0, 0, 0),
		direction(0, 0, 0),
		tMax(RAY_T_MAX)
	{

	}

	Ray(const Ray& r)
	{
		origin = r.origin;
		direction = r.direction;
		tMax = r.tMax;
	}

	Ray(const vec3& origin, const vec3& direction, float max = RAY_T_MAX) :
		origin(origin), direction(direction), tMax(max)
	{

	}

	~Ray()
	{

	}

	Ray& operator =(const Ray& r)
	{
		//TODO implement
		origin = r.origin;
		direction = r.direction;
		tMax = r.tMax;

		return *this;
	}

	vec3 getPoint(float t) const
	{
		return origin + direction * t;
	}
};

class Shape;
class Scene;
class RendererSystem;
class Light;
class Material;

/// <summary>
/// Represents an intersection 
/// </summary>
class HitInfo
{
public:
	int hit_count = 0;
	
	Ray ray;
	float t;
	Shape* shape = NULL;

	vec3 color;
	float depth;
	vec3 normal;

	vec3 position;

	HitInfo()
	{
		t = RAY_T_MAX;
	}

	HitInfo(const HitInfo& o) :
		ray(o.ray),
		t(o.t),
		shape(o.shape),
		color(o.color),
		depth(o.depth),
		normal(o.normal)
	{

	}

	HitInfo(Ray r) :
		ray(r),
		t(RAY_T_MAX)
	{

	}

	bool doesHit() const
	{
		return shape == NULL;
	}

	/*
	vec3 position() const
	{
		return ray.getPoint(t);
	}
	*/
};


class RuntimeResource : public SHObject
{
	SHObject_Base(RuntimeResource)
	
public:
	
	RuntimeResource()
	{

	}
	
	virtual ~RuntimeResource()
	{
		
	}
};


class Shape : public RuntimeResource
{
	SHObject_Base(Shape)
public:
	Shape()
	{

	}
	
	/// <summary>
	/// Fully calculates and fills in the provided hit info object
	/// </summary>
	/// <param name="hitInfo">The hit info to be filled if there was a hit</param>
	/// <returns>Returns true if there was a hit, false if not</returns>
	virtual bool calculateHit(OUT HitInfo& hitInfo) = 0;

	/// <summary>
	/// Only checks if there was a hit. This in most cases is faster that <see cref="calculateHit(HitInfo& hitInfo)"/>
	/// </summary>
	/// <param name="ray">The ray to use for the check</param>
	/// <returns>Returns true if there was a hit, false if not</returns>
	virtual bool doesHit(const Ray& ray) = 0;
};

#pragma region Shapes

class Plane : public Shape
{
	SHObject_Base(Plane)

	vec3 origin;
	vec3 normal;

public:
	Plane(vec3 point = vec3(0, 0, 0), vec3 n = vec3(0, 0, 0)) :
		origin(point), normal(n)
	{

	}

	bool calculateHit(HitInfo& hitInfo) override
	{
		// First, check if we intersect
		float angle = dot(hitInfo.ray.direction, normal);

		//This is the case when the plane and the normal vector are parallel
		//This can happen when the ray is not touching the plane or it is inside the plane
		//we ignore both
		if (angle == 0.0f)
		{
			return false;
		}

		// Find point of intersection
		float t = dot(origin - hitInfo.ray.origin, normal) / angle;

		if (t <= RAY_T_MIN || t >= hitInfo.t)
		{
			// Outside relevant range
			return false;
		}

		hitInfo.t = t;
		hitInfo.shape = this;
		hitInfo.position = hitInfo.ray.getPoint(hitInfo.t);
		//hitInfo.depth = distance(hitInfo.ray.origin, hitInfo.ray.getPoint(hitInfo.t));
		hitInfo.normal = normal;

		return true;
	};

	bool doesHit(const Ray& ray) override
	{
		// First, check if we intersect
		float angle = dot(ray.direction, normal);

		//This is the case when the plane and the normal vector are parallel
		//This can happen when the ray is not touching the plane or it is inside the plane
		//we ignore both
		if (angle == 0.0f)
		{
			return false;
		}

		// Find point of intersection
		float t = dot(origin - ray.origin, normal) / angle;

		if (t <= RAY_T_MIN || t >= ray.tMax)
		{
			// Outside relevant range
			return false;
		}

		return true;
	};
};

class Sphere : public Shape
{
	SHObject_Base(Sphere)

		//vec3 origin;
		float radius;

public:
	Sphere(float r) :
		radius(r)
	{

	}


	bool calculateHit(HitInfo& hitInfo) override
	{
		// we push the whole thing back to the origin so the calculations are easier
		Ray localRay = hitInfo.ray;
		//localRay.origin = localRay.origin - origin;

		float a = length2(localRay.direction);
		float b = 2 * dot(localRay.direction, localRay.origin);
		float c = length2(localRay.origin) - pow2(radius);

		float discr = b * b - 4 * a * c;
		if (discr < 0) return false;

		discr = sqrtf(discr);

		float t1 = (-b - discr) / 2 / a;
		float t2 = (-b + discr) / 2 / a;

		// First check if close intersection is valid
		if (t1 > RAY_T_MIN && t1 < hitInfo.t)
		{
			hitInfo.t = t1;
		}
		else if (t2 > RAY_T_MIN && t2 < hitInfo.t)
		{
			hitInfo.t = t2;
		}
		else
		{
			// Neither is valid
			return false;
		}

		// Finish populating intersection
		hitInfo.shape = this;
		//shitInfo.color = vec3(1,0,0);
		hitInfo.position = hitInfo.ray.getPoint(hitInfo.t);
		hitInfo.depth = hitInfo.t;
		hitInfo.normal = hitInfo.position;

		return true;
	};

	bool doesHit(const Ray& ray) override
	{
		// we push the whole thing back to the origin so the calculations are easier
		Ray localRay = ray;
		//localRay.origin = localRay.origin - origin;

		float a = length2(localRay.direction);
		if (a > 1 - RAY_T_MIN)
			a = 1;
		float b = 2 * dot(localRay.direction, localRay.origin);
		float c = length2(localRay.origin) - pow2(radius);

		float discr = b * b - 4 * a * c;
		if (discr < 0) {
			return false;
		}

		discr = sqrtf(discr);

		float t1 = (-b - discr) / 2 / a;
		float t2 = (-b + discr) / 2 / a;

		// First check if close intersection is valid
		if (t1 > RAY_T_MIN && t1 < ray.tMax)
		{
			return true;
		}
		else if (t2 > RAY_T_MIN && t2 < ray.tMax)
		{
			return true;
		}

		return false;
	};
};

class Hyperboloid : public Shape
{
	SHObject_Base(Hyperboloid)

		mat4 A;
	

public:
	Hyperboloid(float k)
	{
		float a = +1;
		float b = +1;
		float c = -1;
		float d = -k;
		
		A.rows[0].x = a;
		A.rows[1].y = b;
		A.rows[2].z = c;
		A.rows[3].w = d;
	}


	bool calculateHit(HitInfo& hitInfo) override
	{
		
		float a = dot(hitInfo.ray.direction,A);
		float b = pow2(hitInfo.position.y) / b2;
		float c = pow2(hitInfo.position.z) / c2;
		
		float r = xa + yb - zc;

		if(r != 1)
		{
			return false;
		}

		// Finish populating intersection
		hitInfo.shape = this;
		//shitInfo.color = vec3(1,0,0);
		hitInfo.position = hitInfo.ray.getPoint(hitInfo.t);
		hitInfo.depth = hitInfo.t;
		hitInfo.normal = hitInfo.position;

		return true;
	};

	bool doesHit(const Ray& ray) override
	{
		return true;
	};
};

#pragma endregion


class Material: public RuntimeResource
{
	vec3 color;

	vec3 kd = vec3(1,1,1);
	vec3 ks;
	vec3 ka;

	float shiny;

	float n;
	
	enum  MaterialType
	{
		Diffuse,
		Mirror
	};

	MaterialType type;
	
public:
	Material(vec3 col, float s) :
		color(col)
	{
		kd = color;
		ks = color;
		ka = color * M_PI; /* vec3(1, 1, 1);*/
		shiny = s;

		type = Diffuse;
	}

	void setMirror(float n)
	{
		type = Mirror;
		this->n = n;
	}

	float Fresnel_R(float dot, float n2)
	{
		float R0 = pow2((n - n2) / (n + n2));
		return R0 + (1 - R0) * (1 - cos(dot));
	}
	
	
	vec3 calculateColor(HitInfo& hit);
};


class Entity : public SHObject
{
	SHObject_Base(Entity)
protected:
	vec3 pos;

public:
	virtual void init(Scene* scene) = 0;


	void setPos(vec3 p)
	{
		pos = p;
	}

	vec3 getPos()
	{
		return pos;
	}
};

class Renderer : public Entity
{
	SHObject_Base(Renderer)

		Shape* shape;
		Material* mat;
public:

	Renderer(Shape* s, Material* m) : shape(s), mat(m)
	{

	}

	void init(Scene* scene) override;

	bool calculateHit(OUT HitInfo& hitInfo)
	{
		Ray og_ray = hitInfo.ray;
		
		hitInfo.ray.origin = hitInfo.ray.origin - pos;

		//see if the ray actually hits us.
		bool res = shape->calculateHit(hitInfo);

		hitInfo.ray = og_ray;
		
		if (res) {
			//hitInfo.t = localHitInfo.t;
			//hitInfo.depth = distance(hitInfo.ray.origin, hitInfo.ray.getPoint(hitInfo.t));
			//hitInfo.normal = localHitInfo.normal;

			
			hitInfo.position = hitInfo.position + pos;

			vec3 color = mat->calculateColor(hitInfo);
			hitInfo.color = color;
		}

		
		

		return res;
	};

	bool doesHit(const Ray& ray)
	{
		// we push the whole thing back to the origin so the calculations are easier
		Ray localRay = ray;
		localRay.origin = localRay.origin - pos;

		return shape->doesHit(localRay);
	};
};

class Camera : public Entity
{
	SHObject_Base(Camera)

		enum CameraType
	{
		Perspective
	};

	vec3 forward;
	vec3 up;
	vec3 right;

	float h, w;

public:

	Camera() :
		forward(0, 0, 1),
		up(0, 1, 0),
		right(1, 0, 0),
		h(0.5f),
		w(0.5f)
	{
		h = 2;
		w = 2;
	}

	void init(Scene* scene) override{};
	
	Ray makeRay(vec2 point) const
	{
		vec3 direction =
			forward + (point.x * w * right) + (point.y * h * up);

		return Ray(pos, normalize(direction));
	}

	void setParameters(float fov, float aspectRatio)
	{
		h = tan(fov);
		w = h * aspectRatio;
	}
};

class Light : public Entity
{
	SHObject_Base(Light)

		vec3 color;
	float intensity;

public:

	void init(Scene* scene) override;;

	Light(vec3 c = vec3(1, 1, 1), float i = 1) :
		color(c), intensity(i)
	{

	}

	vec3 getColor()
	{
		return color;
	}

	float getIntensity()
	{
		return intensity;
	}
};



class Scene
{
public:

	std::vector<Entity*> objects;
	Camera* mainCam;


	void addEntity(Entity* e)
	{
		objects.push_back(e);
		e->init(this);
		if (e->GetTypeId() == Camera::TypeId() && mainCam == nullptr)
		{
			mainCam = (Camera*)e;
		}
	}

	std::vector<Entity*> getObjects() const
	{
		return objects;
	}

	Camera* getMainCam() const
	{
		return mainCam;
	}
};




class Buffer
{
	int width, height;
	vec3* data;

	Texture output;

public:

	Buffer(int width, int height)
		: width(width), height(height)
	{
		data = new vec3[width * height];
	}

	~Buffer()
	{
		delete[] data;
	}

	vec3* getPixel(int x, int y)
	{
		return data + (x + y * width);
	}

	inline void setPixel(int x, int y, vec3 color)
	{
		data[x + y * width] = color;
	}

	size_t getWidth() { return width; }
	size_t getHeight() { return height; }

	void createTexture(int sampling = GL_LINEAR)
	{
		std::vector<vec4> converted;

		for (int i = (width * height) - 1; i >= 0; i--)
		{
			converted.emplace_back(vec4(data[i].x, data[i].y, data[i].z, 0));
		}

		output.create(width, height, converted, sampling);
	}

	void updateTexture()
	{
		std::vector<vec4> converted;

		for (int i = (width * height) - 1; i >= 0; i--)
		{
			converted.emplace_back(vec4(data[i].x, data[i].y, data[i].z, 0));
		}
		
		glBindTexture(GL_TEXTURE_2D, output.textureId);    // binding
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, converted.data()); // To GPU
	}

	Texture& getTexture()
	{
		return output;
	}
};

class RendererSystem;
RendererSystem* RendererSystem_Instance = nullptr;
class RendererSystem
{
	Scene* scene;

	//static RendererSystem* Instance;
	Buffer target = Buffer(windowWidth, windowHeight);
	
	std::vector<Renderer*> renderers;
	std::vector<Light*> lights;

	vec3 ambient;
	float ambientIntensity = 0;
public:

	static RendererSystem& getRenderer()
	{
		return *RendererSystem_Instance;
	}

	RendererSystem(Scene* s) : scene(s)
	{
		if (RendererSystem_Instance != nullptr)
			printf("ERROR: More than one RendererSystem");
		RendererSystem_Instance = this;

		target.createTexture();
	}



	void renderScene()
	{
		if(scene->getMainCam() == nullptr)
		{
			printf("There is no camera registered to the scene. Can't render it");
			return;
		}

		
		consoleHeader(target);

		//std::ofstream myfile;
		//myfile.open("example.txt");

		int startX = 0;
		int startY = 0;

		int endX = target.getWidth();
		int endY = target.getHeight();
		
		for (size_t y = startY; y < endY; y++)
		{
			consoleLineStart(y);
			for (size_t x = startX; x < endX; x++)
			{
				vec2 imagepos = vec2(
					(2.0f * x) / (float)target.getWidth() - 1.0f,
					(-2.0f * y) / (float)target.getHeight() + 1.0f);


				Ray ray = scene->getMainCam()->makeRay(imagepos);

				bool intersects = false;

				HitInfo hit = HitInfo(ray);
				intersects = trace(hit);
				if (intersects == false)
				{
					consoleEmptyPixel();
					target.setPixel(x, y, 0);
				}
				else {
					//float shade = clamp(1-((hit.depth / 4)-0.95f),0.f,1.0f);

					target.setPixel(x, y, hit.color);

					//myfile << "x: " << x << " y: " << y << " color: " << hit.color.x << ", " << hit.color.y << ", " << hit.color.z << "\n";

					consolePixel(hit.color.x);

				}
			}
			consoleLineEnd(y);
		}

		//myfile.close();
	}

	Texture& display()
	{
		
		target.updateTexture();
		return target.getTexture();
	}

	

	bool trace(HitInfo& hit)
	{
		if (hit.hit_count >= TRACE_MAX_ITERATION)
			return false;
		
		bool intersects = false;
		for (Renderer * obj : renderers)
		{
			auto renderer = ((Renderer*)obj);

			if (renderer->doesHit(hit.ray))
			{
				intersects = true;

				renderer->calculateHit(hit);

			}
		}
		return intersects;
	}

	bool intersectTrace(Ray& ray) const
	{
		bool intersects = false;
		for (Renderer * obj : renderers)
		{
			auto renderer = ((Renderer*)obj);

			if (renderer->doesHit(ray))
			{
				return true;
			}
		}
		return intersects;
	}


	
	void addRenderer(Renderer* r)
	{
		renderers.emplace_back(r);
	}

	void addLight(Light* r)
	{
		lights.emplace_back(r);
	}

	std::vector<Light*>const& getLights()
	{
		return lights;
	}

	void setAmbientLight(vec3 color, float i)
	{
		ambient = color;
		ambientIntensity = i;
	}

	vec3 getAmbientLight()
	{
		return ambient;
	}

	float getAmbinetIntensity()
	{
		return ambientIntensity;
	}
	
	//Consloe debugger
	//#######################################################
#pragma region Consloe Debugger

	void consolePixel(float a)
	{
		if (CONSOLE_RENDER) {
			if (a < 0.1)
			{
				printf(".");
			}
			else if (a < 0.5)
			{
				printf("\\");
			}
			else if (a < 0.8)
			{
				printf("%%");
			}
			else if (a < 1)
			{
				printf("#");
			}
			else
			{
				printf("▓");
			}
		}
	}

	void consoleEmptyPixel()
	{
		if (CONSOLE_RENDER) {
			printf(" ");
		}
	}

	void consoleHeader(Buffer& target)
	{
		if (CONSOLE_RENDER) {
			printf("#");
			for (size_t i = 0; i < target.getWidth(); i++)
			{
				printf("-");
			}
			printf("#\n");
		}
	}

	void consoleLineStart(int i)
	{
		if (CONSOLE_RENDER) {
			printf("|");
		}
	}

	void consoleLineEnd(int i)
	{
		if (CONSOLE_RENDER) {
			printf("|");
		}
	}

#pragma endregion
	//#######################################################
};



#pragma region Late function dec

void Renderer::init(Scene* scene)
{
	RendererSystem::getRenderer().addRenderer(this);
}

void Light::init(Scene* scene)
{
	RendererSystem::getRenderer().addLight(this);
}

vec3 Material::calculateColor(HitInfo& hit)
{
	//float shade = clamp(1 - ((hit.depth / 4) - 0.95f), 0.f, 1.0f);

	RendererSystem* rendererSystem = RendererSystem_Instance;

	vec3 out_color(0,0,0);
	//if(type == Diffuse)
	out_color = (ka * rendererSystem->getAmbientLight()* rendererSystem->getAmbinetIntensity());
	
	for (Light* light : rendererSystem->getLights())
	{
		vec3 target = light->getPos();
		Ray shadowRay = Ray(hit.position + hit.normal * RAY_T_MIN, normalize(target - hit.position));
		shadowRay.tMax = length(target - hit.position);
		float cosTheta = dot(hit.normal, normalize(target - hit.position));
		if (cosTheta > 0 && !rendererSystem->intersectTrace(shadowRay))
		{
			vec3 halfVector = normalize(hit.ray.direction + shadowRay.direction);

			vec3 kd_part(0, 0, 0);
			 kd_part = kd * clamp(dot(shadowRay.direction, hit.normal),0.0f,1.0f);

			vec3 ks_part(0,0,0);
			if(shiny > 0)
			ks_part = ks * pow( clamp(dot(halfVector, hit.normal),0.0f,1.0f),shiny);

			
			//if(type == Diffuse)
			out_color = out_color + light->getColor() * light->getIntensity() * ( kd_part + ks_part);
		}
	}

	if(type == Mirror)
	{
		vec3 reflectionDir = reflect(hit.ray.direction, hit.normal);
		Ray reflectionRay(hit.position + hit.normal * RAY_T_MIN, reflectionDir);
		HitInfo reflectionHit(reflectionRay);
		reflectionHit.hit_count = hit.hit_count+1;
		if(rendererSystem->trace(reflectionHit))
		{
			out_color = out_color + reflectionHit.color *Fresnel_R(dot(hit.ray.direction, hit.normal), 1);
		}
	}

	if(out_color == vec3(0.0f,0.0f,0.0f))
	{
		printf("?? \n");
	}

	//out_color = hit.position;
	
	return out_color;
}

#pragma endregion


//void loadScene(Scene* scene);

class Engine
{
	Scene* scene;
	RendererSystem* renderer;

	std::vector<RuntimeResource*> runtimeResources;
public:
	
	Engine()
	{
		scene = new Scene();
		
		renderer = new RendererSystem(scene);
		loadScene(scene);
	}

	~Engine()
	{
		delete renderer;
		delete scene;

		for (auto var : runtimeResources)
		{
			delete var;
		}
	}

	void Tick()
	{
		
	}

	void Render()
	{
		long timeStart = glutGet(GLUT_ELAPSED_TIME);
		
		renderer->renderScene();
		renderer->display();

		long timeEnd = glutGet(GLUT_ELAPSED_TIME);
		printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));
	}

	RendererSystem* getRenderer()
	{
		return renderer;
	}


	void addRuntimeResource(RuntimeResource* res)
	{
		runtimeResources.push_back(res);
	}

	template<typename T, class ...ARGS>
	T* CreateRuntimeResource(ARGS&&... args)
	{
		T* val = new T(args...);
		addRuntimeResource(val);
		return val;
	}

	
	void loadScene(Scene* scene)
	{
		//Ceate the materials
		auto floor = CreateRuntimeResource<Material>(vec3(0.9, 0.9, 1), 0);
		auto red = CreateRuntimeResource<Material>(vec3(1, 0, 0), 0);
		red->setMirror(4.7f);

		auto blue = CreateRuntimeResource<Material>(vec3(66 / 255.0f, 239 / 255.0f, 245 / 255.0f), 700);

		
		//Scene scene;
		//RendererSystem renerer(&scene);
		renderer->setAmbientLight(vec3(255 / 255.0f, 248 / 255.0f, 168 / 255.0f), 0.1);

		auto plane = CreateRuntimeResource<Plane>(vec3(0, 0, 0), CoordinateSystem::Up);
		auto ground = new Renderer(plane, floor);
		scene->addEntity(ground);

		auto side_plane = CreateRuntimeResource<Plane>(vec3(6, 0, 0), -CoordinateSystem::Right);
		auto side1 = new Renderer(side_plane, floor);
		scene->addEntity(side1);

		auto sphere = CreateRuntimeResource<Sphere>(1);

		auto test = new Renderer(sphere, red);
		test->setPos(vec3(0.5, 2.3, 13));
		scene->addEntity(test);

		auto test2 = new Renderer(sphere, blue);
		test2->setPos(vec3(2, 1, 10));
		scene->addEntity(test2);


		auto pointLight = new Light(vec3(1, 1, 1), 0.5);
		pointLight->setPos(vec3(2, 5, 2));
		scene->addEntity(pointLight);


		Camera* cam = new Camera();
		cam->setPos(vec3(0, 2, -5));
		cam->setParameters(25.0f * M_PI / 180.0f,
			(float)windowWidth / (float)windowHeight);
		scene->addEntity(cam);

	}
};








#pragma region Shaders

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

#pragma endregion

GPUProgram gpuProgram; // vertex and fragment shaders

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	//Texture texture;
public:
	FullScreenTexturedQuad(/*int windowWidth, int windowHeight , std::vector<vec4>& image*/)
		//: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw(Texture& texture) {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;




Engine* engine;
// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	engine = new Engine();
	

	//renerer.renderScene(target);

	//target.createTexture(output);

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad();

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	engine->Render();
	fullScreenTexturedQuad->Draw(engine->getRenderer()->display());
	glutSwapBuffers();					// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	glutPostRedisplay();
}