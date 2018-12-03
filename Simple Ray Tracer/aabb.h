#ifndef AABB
#define AABB

#include "hitable.h"
#include "ray.h"


inline float ffmin(float a, float b) { return a < b ? a : b; }
inline float ffmax(float a, float b) { return a > b ? a : b; }

float ffmin(__m128 a, __m128 b) 
{ 
	float res_a[4];
	_mm_store1_ps(&res_a[0], a);

	float res_b[4];
	_mm_store1_ps(&res_b[0], b);

	return res_a[0] < res_b[0] ? res_a[0] : res_b[0];

}
float ffmax(__m128 a, __m128 b)
{
	float res_a[4];
	_mm_store1_ps(&res_a[0], a);

	float res_b[4];
	_mm_store1_ps(&res_b[0], b);

	return res_a[0] > res_b[0] ? res_a[0] : res_b[0]; 
}

class aabb
{
public:
	aabb() {}
	aabb(const vec3& a, const vec3& b) { _min = a; _max = b; }

	vec3 min() const { return _min; }
	vec3 max() const { return _max; }

	bool hit(const ray& r, float tmin, float tmax) const {
		for (size_t i = 0; i < 3; i++)
		{
			float t0 = ffmin((_min[i] - r.origin()[i]) / r.direction()[i], (_max[i] - r.origin()[i]) / r.direction()[i] );
			float t1 = ffmax((_min[i] - r.origin()[i]) / r.direction()[i], (_max[i] - r.origin()[i]) / r.direction()[i]);

			tmin = ffmax(t0, tmin);
			tmax = ffmin(t1, tmax);

			if (tmax <= tmin) return false;
		}

		return true;
	}


private:
	vec3 _min;
	vec3 _max;
};

aabb surrounding_box(aabb box0, aabb box1)
{
	vec3 small(
		ffmin(box0.min().x(), box1.min().x()),
		ffmin(box0.min().y(), box1.min().y()),
		ffmin(box0.min().z(), box1.min().z())
	);

	vec3 big(
		ffmax(box0.max().x(), box1.max().x()),
		ffmax(box0.max().y(), box1.max().y()),
		ffmax(box0.max().z(), box1.max().z())
	);

	return aabb(small, big);

}

#endif