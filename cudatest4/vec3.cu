
#include "vec3.h"

__host__ __device__ vec3::vec3() { }

__host__ __device__ vec3::vec3(float x, float y, float z) : x(x), y(y), z(z) { }

__host__ __device__ vec3 operator + (float a, const vec3& v) {
	return { v.x + a, v.y + a, v.z + a };
}

__host__ __device__ vec3 operator + (const vec3& v, float a) {
	return { v.x + a, v.y + a, v.z + a };
}

__host__ __device__ vec3 operator - (const vec3& v, float a) {
	return { v.x - a, v.y - a, v.z - a };
}

__host__ __device__ vec3 operator * (const vec3& v, float a) {
	return { v.x * a, v.y * a, v.z * a };
}

__host__ __device__ vec3 operator * (float a, const vec3& v) {
	return { v.x * a, v.y * a, v.z * a };
}

__host__ __device__ vec3 operator / (const vec3& v, float a) {
	return { v.x / a, v.y / a, v.z / a };
}

__host__ __device__ vec3& operator += (vec3& v, float a) {
	v = v + a;
	return v;
}

__host__ __device__ vec3& operator -= (vec3& v, float a) {
	v = v - a;
	return v;
}

__host__ __device__ vec3& operator *= (vec3& v, float a) {
	v = v * a;
	return v;
}

__host__ __device__ vec3& operator /= (vec3& v, float a) {
	v = v / a;
	return v;
}

__host__ __device__ vec3 operator + (const vec3& v1, const vec3& v2) {
	return vec3{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
}

__host__ __device__ vec3 operator - (const vec3& v1, const vec3& v2) {
	return vec3{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
}

__host__ __device__ vec3 operator * (const vec3& v1, const vec3& v2) {
	return vec3{ v1.x * v2.x, v1.y * v2.y, v1.z * v2.z };
}

__host__ __device__ vec3 operator / (const vec3& v1, const vec3& v2) {
	return vec3{ v1.x / v2.x, v1.y / v2.y, v1.z / v2.z };
}

__host__ __device__ vec3& operator += (vec3& v1, const vec3& v2) {
	v1 = v1 + v2;
	return v1;
}

__host__ __device__ vec3& operator -= (vec3& v1, const vec3& v2) {
	v1 = v1 - v2;
	return v1;
}

__host__ __device__ vec3& operator *= (vec3& v1, const vec3& v2) {
	v1 = v1 * v2;
	return v1;
}

__host__ __device__ vec3& operator /= (vec3& v1, const vec3& v2) {
	v1 = v1 / v2;
	return v1;
}

__host__ __device__ float dot(const vec3& a, const vec3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ vec3 reflect(const vec3& I, const vec3& N) {
	return I - 2.0f * dot(N, I) * N;
}

__host__ __device__ vec3 mod(const vec3& v, float n) {
	return v - n * floor(v / n);
}

__host__ __device__ vec3 floor(const vec3& v) {
	return {
		floor(v.x),
		floor(v.y),
		floor(v.z)
	};
}

__host__ __device__ vec3 absolute(const vec3& v) {
	return {
		abs(v.x),
		abs(v.y),
		abs(v.z)
	};
}

__host__ __device__ vec3 maximum(const vec3& v, float a) {
	return vec3{
		max(v.x, a),
		max(v.y, a),
		max(v.z, a)
	};
}

__host__ __device__ float length(const vec3& v) {
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ vec3 normalize(const vec3& v) {
	float l = length(v);
	return {
		v.x / l,
		v.y / l,
		v.z / l
	};
}
/*
__host__ __device__ vec3 vec3::rotateX(float theta) {
	mat3 m(
		1, 0, 0,
		0, cos(theta), -sin(theta),
		0, sin(theta), cos(theta)
	);
	return *this * m;
}

__host__ __device__ vec3 vec3::rotateY(float theta) {
	mat3 m(
		cos(theta), 0, sin(theta),
		0, 1, 0,
		-sin(theta), 0, cos(theta)
	);
	return *this * m;
}

__host__ __device__ vec3 vec3::rotateZ(float theta) {
	mat3 m(
		cos(theta), -sin(theta), 0,
		sin(theta), cos(theta), 0,
		0, 0, 1
	);
	return *this * m;
}
*/
string vec3::toString() {
	ostringstream oss;
	oss << "(" << x << ", " << y << ", " << z << ")";
	return oss.str();
}

ostream& operator << (ostream& s, vec3& v) {
	s << v.toString();
	return s;
}