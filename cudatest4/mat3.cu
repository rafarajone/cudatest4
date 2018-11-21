
#include "mat3.h"

__host__ __device__ mat3::mat3() {}

__host__ __device__ mat3::mat3(float e0, float e1, float e2, float e3, float e4, float e5, float e6, float e7, float e8) :
	e0(e0), e1(e1), e2(e2), e3(e3), e4(e4), e5(e5), e6(e6), e7(e7), e8(e8) {}

__host__ __device__ vec3 operator * (const vec3& v, const mat3& m) {
	return {
		v.x * m.e0 + v.y * m.e1 + v.z * m.e2,
		v.x * m.e3 + v.y * m.e4 + v.z * m.e5,
		v.x * m.e6 + v.y * m.e7 + v.z * m.e8
	};
}

string mat3::toString() {
	ostringstream oss;
	oss << e0 << " " << e1 << " " << e2 << "\n"
		<< e3 << " " << e4 << " " << e5 << "\n"
		<< e6 << " " << e7 << " " << e8;
	return oss.str();
}