#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <string.h>
/* OMP */
#include <omp.h>
/* CUDA */
#include <cuda.h>
/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>
/* 3D vector manipulation abstraction */
#include <vec_gpu.h>

#ifndef MAX
#define MAX(x,y) ((x)>(y)?(x):(y))
#endif

#ifndef PI
#define PI 3.1415926535897
#endif

/** CUDA Error Handling Macros */
#define CUDA_ERROR_CHECK
#define CudaCheckError()	__cudaCheckError( __FILE__, __LINE__ )
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
	    printf("Error at %s:%d\n",__FILE__,__LINE__); \
	    return EXIT_FAILURE;}} while(0)





/** Window and view point */
typedef struct __Camera {
	Vec3 pos;
	double height, width;
} Camera;

/** CUDA Error Handling Util */
void __cudaCheckError( const char *file, const int line );

/** Util */
void save_grid(double * data, int px, int py, const char * fname);
void setArgs(int argc, char** argv);

/** Ray Tracer */
__global__ void setup_seed(curandState_t *state, unsigned long seed);
__global__ void rayTrace(int px, int py, int ray_per_task, Camera camera, double * mat_grid, curandState_t *globalStates);
__device__ void randomDirection(Vec3 * vptr, curandState_t *globalStates);
__device__ void rand_double(double start, double end, double *result, curandState_t *globalStates);

/** Global variables */
int num_pixel;
int num_block;
int num_ray;
int num_thread;





/**************************************************
*
*	Driver
*
**************************************************/

int main(int argc, char ** argv) {

	num_pixel = 1000;
	num_ray = 1e6;
	num_block = 1;
	num_thread = 100;
	Camera cam;

	cam.height = 20.0;
	cam.width = 20.0;

	setArgs(argc, argv);
	int ray_per_task = num_ray / (num_block * num_thread);

	/** Print GPU Device Name */
	cudaDeviceProp prop;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);
	printf("%-25s%s\n", "CUDA Device: ", prop.name);

	/** Allocate serialized Grid Array on Host & Device */
	double * host_grid = (double*) malloc(num_pixel * num_pixel * sizeof(double));
	double * device_grid;
	CUDA_CALL(cudaMalloc((void**) &device_grid, num_pixel*num_pixel*sizeof(double)));
	CUDA_CALL(cudaMemcpy(device_grid, host_grid, num_pixel*num_pixel * sizeof(double), cudaMemcpyHostToDevice));

	/** Allocate state for random seed */
	curandState_t *states;
	CUDA_CALL(cudaMalloc((void**) &states, num_block * num_thread * sizeof(curandState_t)));

	/** Setup seeds */
	setup_seed<<<num_block, num_thread>>>(states, time(NULL));

	/** CUDA timer variables */
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float dt = 0;

	/** CUDA jobs */
	cudaEventRecord(start, 0);
	/** Do something */
	rayTrace<<<num_block, num_thread>>>(num_pixel, num_pixel, ray_per_task cam, device_grid, states);
	/** Complete something */
	CudaCheckError();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&dt, start, stop);
	cudaDeviceSynchronize();

	CUDA_CALL(cudaMemcpy(host_grid, device_grid, num_pixel*num_pixel * sizeof(double), cudaMemcpyDeviceToHost));

	save_grid(host_grid, num_pixel, num_pixel, "output.gpu.out");

	printf("%d\t%d\t%lf\n", num_pixel*num_pixel, dt/1000.0);

	free(host_grid);
	CUDA_CALL(cudaFree(device_grid));
}





/**************************************************
*
*	Ray tracer functions called on GPU by Host
*
**************************************************/

__global__ void setup_seed(curandState_t *state, unsigned long seed) {

	/* Initialize the state */

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
		idx, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&state[idx]);
}

__global__ void rayTrace(int px, int py, int ray_per_task, Camera camera,
	double * mat_grid, curandState_t *globalStates) {

	int n;

	/** Model parameters */
	double radius = 6.0;
	Vec3 vec_c; // position of sphere centre
	Vec3 vec_l; // position of light source

	/** Ray tracer variables global */
	double dotp_cc;

	/** Window display */
	double w_max_x, w_min_x, w_max_z, w_min_z;

	/** Initialization */
	vec3(0.0, 12.0, 0.0, &vec_c);
	vec3(4.0, 4.0, -1.0, &vec_l);
	vec3(0.0, 10.0, 0.0, &camera.pos);
	w_max_x = camera.pos.x + camera.width * 0.5;
	w_min_x = camera.pos.x - camera.width * 0.5;
	w_max_z = camera.pos.z + camera.height * 0.5;
	w_min_z = camera.pos.z - camera.height * 0.5;
	vec3DotP(&vec_c, &vec_c, &dotp_cc);

	for (n=0; n<ray_per_task; n++) {

		/** Ray tracer variables local */
		int i, j;
		double delta, solution, brightness;
		double dotp_vc;
		Vec3 vec_v; // view ray vector
		Vec3 vec_i; // position of intersection
		Vec3 vec_s; // direction of light source at I
		Vec3 vec_n; // unit normal vector at I
		Vec3 vec_w; // camera vector

		do { // sample random V from unit sphere
			do {
				randomDirection(&vec_v, globalStates);
				vec3Scale(&vec_v, camera.pos.y / vec_v.y, &vec_w);
			} while (vec_w.x < w_min_x || vec_w.x > w_max_x || vec_w.z < w_min_z || vec_w.z > w_max_z);
			vec3DotP(&vec_v, &vec_c, &dotp_vc);
			delta = dotp_vc*dotp_vc + radius*radius - dotp_cc;
		} while (delta < 0); // delta > 0, enable to find an intersection

		solution = dotp_vc - sqrtf(delta);

		vec3Scale(&vec_v, solution, &vec_i);
		vec3Combine(&vec_i, &vec_c, 1.0, -1.0, &vec_n);
		vec3Normalize(&vec_n);
		vec3Combine(&vec_l, &vec_i, 1.0, -1.0, &vec_s);
		vec3Normalize(&vec_s);

		vec3DotP(&vec_s, &vec_n, &brightness);
		brightness = MAX(brightness,0);

		j = px - 1 - (int) ((double) px * (vec_w.x - w_min_x) / (camera.width));
		i = (int) ((double) py * (vec_w.z - w_min_z) / (camera.height));

		//#pragma omp atomic update
		mat_grid[i * px + j] += brightness;
		//atomicAdd(mat_grid[i * px + j], brightness);
	}
}





/**************************************************
*
*	Ray tracer functions called on GPU by GPU
*
**************************************************/

__device__ void randomDirection(Vec3 * vptr, curandState_t *globalStates) {

	/*
	* return a unit vector of random direction.
	*/

	double angle_psi;
	double angle_theta;

	rand_double(0, 2 * PI, &angle_psi, globalStates);
	rand_double(0, PI, &angle_theta, globalStates);

	vptr->x = sin(angle_theta) * cos(angle_psi);
	vptr->y = sin(angle_theta) * sin(angle_psi);
	vptr->z = cos(angle_theta);
}

__device__ void rand_double(double start, double end, double *result, curandState_t *globalStates) {

	/*
	* return a random double between start and end.
	*/

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curandState_t *localState = &globalStates[idx];

	/* curand works like rand - except that it takes a state as a parameter */
	double r = curand_uniform(localState);
	double x = (start < end) ? (start) : (end);
	double y = (start < end) ? (end) : (start);
	*result = (y - x) * r + x;
}





/**************************************************
*
*	Util functions called on Host
*
**************************************************/

void __cudaCheckError(const char *file, const int line) {

	#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
	#endif
}

void save_grid(double * data, int px, int py, const char * fname) {

	FILE * fp = fopen(fname, "w");
	int i, j;

	for (i=0; i<py; i++) {
		for (j=0; j<px; j++) {
			fprintf(fp, "%lf ", data[i * px + j]);
		} fprintf(fp, "\n");
	}

	fclose(fp);
}

void setArgs(int argc, char** argv) {

	/*
	* getopt_long stores the option index here.
	*/

	int option_index = 0;
	int ch;

	static struct option long_options[] = {
		//{"abc", 0|no_argument|required_argument|optional_argument, flag, 'a'},
		{"pixel", required_argument, 0, 'n'},
		{"ray", required_argument, 0, 'r'},
		{"block", required_argument, 0, 'b'},
		{"thread", required_argument, 0, 't'},
		{0, 0, 0, 0}
	};

	/* Detect the end of the options. */
	while ( (ch = getopt_long(argc, argv, "n:r:b:t:", long_options, &option_index)) != -1 ) {
		switch (ch) {
			case 'n':
				num_pixel = atoi(optarg);
				break;
			case 'r':
				num_ray = atoi(optarg);
				break;
			case 'b':
				num_block = atoi(optarg);
				break;
			case 't':
				num_thread = atoi(optarg);
				break;
			case '?':
				printf("Unknown option\n");
				break;
			case 0:
				break;
		}
	}
}
