//3D vector class for different operations.

// Right now supports SSE operations. 
// Maybe SSE is not perfectly used here, 
// because I wanted to replace the existing vec3 functionalities with minimal dependant code changes

#ifndef VEC3H
#define VEC3H

#include <math.h>
#include <stdlib.h>
#include <iostream>

#include <xmmintrin.h>

class vec3
{
public:
	vec3() {}


//#ifdef SSE

	vec3(float e0, float e1, float e2)
	{
		sse_vec3.x = _mm_set1_ps(e0);
		sse_vec3.y = _mm_set1_ps(e1);
		sse_vec3.z = _mm_set1_ps(e2);
	}
	vec3(__m128 e0, __m128 e1, __m128 e2)
	{
		sse_vec3.x = e0;
		sse_vec3.y = e1;
		sse_vec3.z = e2;
	}

	inline __m128 x() const { return sse_vec3.x; }
	inline __m128 y() const { return sse_vec3.y; }
	inline __m128 z() const { return sse_vec3.z; }

	inline __m128 r() const { return sse_vec3.x; }
	inline __m128 g() const { return sse_vec3.y; }
	inline __m128 b() const { return sse_vec3.z; }


	inline __m128 operator[] (int i) const
	{
		switch (i)
		{
		case 0:
			return sse_vec3.x;
		case 1:
			return sse_vec3.y;
		case 2:
			return sse_vec3.z;
		}
	}

	inline const vec3& operator+() const { return *this; }
	inline vec3 operator-() const
	{ 
		float res0[4];
		float res1[4];
		float res2[4];


		_mm_store1_ps(&res0[0], _mm_sub_ps(_mm_set1_ps(0.0), sse_vec3.x));
		_mm_store1_ps(&res1[0], _mm_sub_ps(_mm_set1_ps(0.0), sse_vec3.y));
		_mm_store1_ps(&res2[0], _mm_sub_ps(_mm_set1_ps(0.0), sse_vec3.z));


		return vec3(res0[0], res1[0], res2[0]);

		//return vec3(0, 0, 0);
	}
	inline vec3& operator+=(const vec3 &v2);
	inline vec3& operator-=(const vec3 &v2);
	inline vec3& operator*=(const vec3 &v2);
	inline vec3& operator/=(const vec3 &v2);
	inline vec3& operator*=(const float t);
	inline vec3& operator/=(const float t);

	inline __m128 length() const
	{
		__m128 x2 = _mm_mul_ps(sse_vec3.x, sse_vec3.x);
		__m128 y2 = _mm_mul_ps(sse_vec3.y, sse_vec3.y);
		__m128 z2 = _mm_mul_ps(sse_vec3.z, sse_vec3.z);
	
		__m128 len = _mm_add_ps(x2, y2);
		len = _mm_add_ps(len, z2);
		len = _mm_sqrt_ps(len);
	
		return len;
	}
	inline __m128 squared_length() const
	{

		__m128 ret;
		SSE_VEC3 tmp;

		tmp.x = _mm_mul_ps(sse_vec3.x, sse_vec3.x);
		tmp.y = _mm_mul_ps(sse_vec3.y, sse_vec3.y);
		tmp.z = _mm_mul_ps(sse_vec3.z, sse_vec3.z);
		ret = _mm_add_ps(tmp.x, tmp.y);
		ret = _mm_add_ps(ret, tmp.z);

		return ret;
	}

	inline void make_unit_vector();

	inline static float ToFloat(__m128 a);

	struct SSE_VEC3
	{
		__m128 x;
		__m128 y;
		__m128 z;
	};

	SSE_VEC3 sse_vec3;

//#endif // SSE


};


//#ifdef SSE

inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
	
	__m128 x = _mm_add_ps(v1.x(), v2.x());
	__m128 y = _mm_add_ps(v1.y(), v2.y());
	__m128 z = _mm_add_ps(v1.z(), v2.z());

	return vec3(x, y, z);

}
inline __m128 operator+(const __m128 v1, const __m128 v2)
{
	return _mm_add_ps(v1, v2);
}
inline vec3 operator-(const vec3 &v1, const vec3 &v2) {

	__m128 x = _mm_sub_ps(v1.x(), v2.x());
	__m128 y = _mm_sub_ps(v1.y(), v2.y());
	__m128 z = _mm_sub_ps(v1.z(), v2.z());

	return vec3(x, y, z);

}
inline __m128 operator-(const __m128 v1, const __m128 v2)
{
	return _mm_sub_ps(v1, v2);
}
inline vec3 operator*(const vec3 &v1, const vec3 &v2) {

	__m128 x = _mm_mul_ps(v1.x(), v2.x());
	__m128 y = _mm_mul_ps(v1.y(), v2.y());
	__m128 z = _mm_mul_ps(v1.z(), v2.z());

	return vec3(x, y, z);

}
inline vec3 operator*(float t, const vec3 &v2) {

	const __m128 scalar = _mm_set1_ps(t);

	__m128 x = _mm_mul_ps(scalar, v2.x());
	__m128 y = _mm_mul_ps(scalar, v2.y());
	__m128 z = _mm_mul_ps(scalar, v2.z());

	return vec3(x, y, z);

}
inline vec3 operator*(const vec3 &v2, float t) {

	const __m128 scalar = _mm_set1_ps(t);

	__m128 x = _mm_mul_ps(scalar, v2.x());
	__m128 y = _mm_mul_ps(scalar, v2.y());
	__m128 z = _mm_mul_ps(scalar, v2.z());

	return vec3(x, y, z);

}
inline __m128 operator*(float t, const __m128 v)
{
	const __m128 scalar = _mm_set1_ps(t);

	__m128 res = _mm_mul_ps(scalar, v);

	return res;
}
inline __m128 operator*(const __m128 v1, const __m128 v2) {

	return _mm_mul_ps(v1, v2);

}
inline vec3 operator/(const vec3 &v1, const vec3 &v2) {

	__m128 x = _mm_div_ps(v1.x(), v2.x());
	__m128 y = _mm_div_ps(v1.y(), v2.y());
	__m128 z = _mm_div_ps(v1.z(), v2.z());

	return vec3(x, y, z);
}
inline vec3 operator/(vec3 v2, float t) {

	const __m128 scalar = _mm_set1_ps(t);

	__m128 x = _mm_div_ps(v2.x(), scalar);
	__m128 y = _mm_div_ps(v2.y(), scalar);
	__m128 z = _mm_div_ps(v2.z(), scalar);

	return vec3(x, y, z);

}
inline __m128 operator/(const __m128 v1, const __m128 v2) {

	return _mm_div_ps(v1, v2);

}
inline vec3& vec3::operator+=(const vec3 &v) {

	sse_vec3.x = _mm_add_ps(sse_vec3.x, v.x());
	sse_vec3.y = _mm_add_ps(sse_vec3.y, v.y());
	sse_vec3.z = _mm_add_ps(sse_vec3.z, v.z());
	return *this;
}
inline vec3& vec3::operator*=(const vec3 &v) {
	sse_vec3.x = _mm_mul_ps(sse_vec3.x, v.x());
	sse_vec3.y = _mm_mul_ps(sse_vec3.y, v.y());
	sse_vec3.z = _mm_mul_ps(sse_vec3.z, v.z());
	return *this;
}
inline vec3& vec3::operator/=(const vec3 &v) {
	sse_vec3.x = _mm_div_ps(sse_vec3.x, v.x());
	sse_vec3.y = _mm_div_ps(sse_vec3.y, v.y());
	sse_vec3.z = _mm_div_ps(sse_vec3.z, v.z());
	return *this;
}
inline vec3& vec3::operator-=(const vec3& v) {
	sse_vec3.x = _mm_sub_ps(sse_vec3.x, v.x());
	sse_vec3.y = _mm_sub_ps(sse_vec3.y, v.y());
	sse_vec3.z = _mm_sub_ps(sse_vec3.z, v.z());
	return *this;
}
inline vec3& vec3::operator*=(const float t) {
	const __m128 scalar = _mm_set1_ps(t);

	sse_vec3.x = _mm_mul_ps(sse_vec3.x, scalar);
	sse_vec3.y = _mm_mul_ps(sse_vec3.y, scalar);
	sse_vec3.z = _mm_mul_ps(sse_vec3.z, scalar);
	return *this;
}
inline vec3& vec3::operator/=(const float t) {
	const __m128 scalar = _mm_set1_ps(1.0/t);

	sse_vec3.x = _mm_mul_ps(sse_vec3.x, scalar);
	sse_vec3.y = _mm_mul_ps(sse_vec3.y, scalar);
	sse_vec3.z = _mm_mul_ps(sse_vec3.z, scalar);
	return *this;
}

inline __m128 sqrt(__m128 v)
{
	return _mm_sqrt_ps(v);
}
inline __m128 dot(const vec3 &v1, const vec3 &v2) 
{

	__m128 x2 = _mm_mul_ps(v1.x(), v2.x());
	__m128 y2 = _mm_mul_ps(v1.y(), v2.y());
	__m128 z2 = _mm_mul_ps(v1.z(), v2.z());

	__m128 res = _mm_add_ps(x2, y2);
	res = _mm_add_ps(res, z2);

	return res;
}
inline vec3 cross(const vec3 &v1, const vec3 &v2) {

	__m128 a = _mm_sub_ps(_mm_mul_ps(v1.y(), v2.z()), _mm_mul_ps(v1.z(), v2.y()));
	__m128 b = _mm_sub_ps(_mm_mul_ps(v1.z(), v2.x()), _mm_mul_ps(v1.x(), v2.z()));
	__m128 c = _mm_sub_ps(_mm_mul_ps(v1.x(), v2.y()), _mm_mul_ps(v1.y(), v2.x()));

	return vec3(a,b,c);
}

inline vec3 unit_vector(vec3 v)
{
	__m128 x2 = _mm_mul_ps(v.x(), v.x());
	__m128 y2 = _mm_mul_ps(v.y(), v.y());
	__m128 z2 = _mm_mul_ps(v.z(), v.z());

	__m128 len = _mm_add_ps(x2, y2);
	len = _mm_add_ps(len, z2);
	len = _mm_sqrt_ps(len);

	__m128 x = _mm_div_ps(v.x(), len);
	__m128 y = _mm_div_ps(v.y(), len);
	__m128 z = _mm_div_ps(v.z(), len);

	return vec3(x, y, z);
}
inline void vec3::make_unit_vector() {
	__m128 ones = _mm_set_ps(1.0f, 1.0f, 1.0f, 1.0f);

	__m128 length = vec3::length();
	
	__m128 k = _mm_div_ps(ones, length);

	sse_vec3.x = _mm_mul_ps(sse_vec3.x, k);
	sse_vec3.y = _mm_mul_ps(sse_vec3.y, k);
	sse_vec3.z = _mm_mul_ps(sse_vec3.z, k);

}
inline float vec3::ToFloat(__m128 a)
{
	float res_a[4];
	_mm_store1_ps(&res_a[0], a);

	return res_a[0];
}


//#endif // SSE

#endif