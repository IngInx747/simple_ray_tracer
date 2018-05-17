#include <vec.h>

/*
* typedef struct _Vec3 {
* 	double x, y, z;
* } Vec3;
*/

Vec3 vec3(double x, double y, double z) {

	/*
	*
	*/

	Vec3 dest;

	dest.x = x;
	dest.y = y;
	dest.z = z;

	return dest;
}

Vec3 vec3Scale(Vec3 vec, double scale) {

	/*
	*
	*/

	Vec3 dest;

	dest.x = vec.x * scale;
	dest.y = vec.y * scale;
	dest.z = vec.z * scale;

	return dest;
}

Vec3 vec3Combine(Vec3 vec1, Vec3 vec2, double alpha, double beta) {

	/*
	*
	*/

	Vec3 dest;

	dest.x = vec1.x * alpha + vec2.x * beta;
	dest.y = vec1.y * alpha + vec2.y * beta;
	dest.z = vec1.z * alpha + vec2.z * beta;

	return dest;
}

double vec3DotP(Vec3 vec1, Vec3 vec2) {

	/*
	*
	*/

	double ans = 0;

	ans = vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;

	return ans;
}
