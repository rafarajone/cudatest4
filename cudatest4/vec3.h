
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>

using namespace std;

struct mat3;

struct vec3 {
	
	float x, y, z;

	__host__ __device__ vec3();
	__host__ __device__ vec3(float, float, float);

	__host__ __device__ friend vec3 operator + (const vec3&, float);
	__host__ __device__ friend vec3 operator + (float, const vec3&);
	__host__ __device__ friend vec3 operator - (const vec3&, float);
	__host__ __device__ friend vec3 operator * (const vec3&, float);
	__host__ __device__ friend vec3 operator * (float, const vec3&);
	__host__ __device__ friend vec3 operator / (const vec3&, float);
	
	__host__ __device__ friend vec3& operator += (vec3&, float);
	__host__ __device__ friend vec3& operator -= (vec3&, float);
	__host__ __device__ friend vec3& operator *= (vec3&, float);
	__host__ __device__ friend vec3& operator /= (vec3&, float);
	
	__host__ __device__ friend vec3 operator + (const vec3&, const vec3&);
	__host__ __device__ friend vec3 operator - (const vec3&, const vec3&);
	__host__ __device__ friend vec3 operator * (const vec3&, const vec3&);
	__host__ __device__ friend vec3 operator / (const vec3&, const vec3&);
	
	__host__ __device__ friend vec3& operator += (vec3&, const vec3&);
	__host__ __device__ friend vec3& operator -= (vec3&, const vec3&);
	__host__ __device__ friend vec3& operator *= (vec3&, const vec3&);
	__host__ __device__ friend vec3& operator /= (vec3&, const vec3&);

	__host__ __device__ friend float dot(const vec3&, const vec3&);
	__host__ __device__ friend vec3 reflect(const vec3&, const vec3&);

	__host__ __device__ friend vec3 mod(const vec3&, float);
	__host__ __device__ friend vec3 floor(const vec3&);
	__host__ __device__ friend vec3 absolute(const vec3&);
	__host__ __device__ friend vec3 maximum(const vec3&, float);

	__host__ __device__ friend float length(const vec3&);
	__host__ __device__ vec3 normalize(const vec3&);
	/*
	__host__ __device__ vec3 rotateX(float);
	__host__ __device__ vec3 rotateY(float);
	__host__ __device__ vec3 rotateZ(float);
	*/
	string toString();
	friend ostream& operator << (ostream& stream, vec3& v1);
};
