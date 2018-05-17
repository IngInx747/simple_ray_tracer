#ifndef VEC_H
#define VEC_H

typedef struct _Vec3 {
	double x, y, z;
} Vec3;

/** Vector assigning */
Vec3 vec3(double x, double y, double z);

/** Vector manipulation */
Vec3 vec3Scale(Vec3 vec, double scale);
Vec3 vec3Combine(Vec3 vec1, Vec3 vec2, double alpha, double beta);

/** Vector degeneration */
double vec3DotP(Vec3, Vec3);

#endif