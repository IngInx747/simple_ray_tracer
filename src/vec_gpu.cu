#include <cuda.h>
#include <vec_gpu.h>

/*
* typedef struct _Vec3 {
* 	double x, y, z;
* } Vec3;
*/


/**************************************************
*
*	VEC3 Util functions called on GPU device
*
**************************************************/

__device__ void vec3(double x, double y, double z, Vec3 *vptr) {

	/*
	*
	*/

	vptr->x = x;
	vptr->y = y;
	vptr->z = z;
}

__device__ void vec3Scale(Vec3 *vec, double scale, Vec3 *vptr) {

	/*
	*
	*/

	if (vec == vptr) {
		vec->x *= scale;
		vec->y *= scale;
		vec->z *= scale;
	} else {
		vptr->x = vec->x * scale;
		vptr->y = vec->y * scale;
		vptr->z = vec->z * scale;
	}
}

__device__ void vec3Combine(Vec3 *vec1, Vec3 *vec2, double alpha, double beta, Vec3 *vptr) {

	/*
	*
	*/

	vptr->x = vec1->x * alpha + vec2->x * beta;
	vptr->y = vec1->y * alpha + vec2->y * beta;
	vptr->z = vec1->z * alpha + vec2->z * beta;
}

__device__ void vec3DotP(Vec3 *vec1, Vec3 *vec2, double *ans) {

	/*
	*
	*/

	*ans = vec1->x * vec2->x + vec1->y * vec2->y + vec1->z * vec2->z;
}

__device__ void vec3Normalize(Vec3 *vptr) {

	/*
	*
	*/

	double scale;

	vec3DotP(vptr, vptr, &scale);

	scale = 1 / sqrtf(scale);

	vptr->x = vptr->x * scale;
	vptr->y = vptr->y * scale;
	vptr->z = vptr->z * scale;
}
