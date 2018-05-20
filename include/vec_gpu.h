#ifndef VEC_GPU_H
#define VEC_GPU_H

typedef struct __Vec3 {
	double x, y, z;
} Vec3;

/** Vector assigning */
__device__ void vec3(double x, double y, double z, Vec3 *vptr);

/** Vector manipulation */
__device__ void vec3Scale(Vec3 *vec, double scale, Vec3 *vptr);

__device__ void vec3Combine(Vec3 *vec1, Vec3 *vec2, double alpha, double beta, Vec3 *vptr);

__device__ void vec3Normalize(Vec3 *vptr);

/** Vector degeneration */
__device__ void vec3DotP(Vec3 *vec1, Vec3 *vec2, double *ans);


#endif