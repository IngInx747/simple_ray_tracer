#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <string.h>
#include <omp.h>
#include <vec.h>

#ifndef MAX
#define MAX(x,y) ((x)>(y)?(x):(y))
#endif

#ifndef PI
#define PI 3.1415926535897
#endif

typedef struct __Camera {
	Vec3 pos;
	double height, width;
} Camera;

/** Ray Tracer */
void rayTrace(int px, int py, int nrays, Camera camera);
Vec3 randomDirection();

/** Util */
double rand_double(double x, double y);
void save_grid(double * data, int px, int py, char * fname);
void setArgs(int argc, char** argv);

/** Global variables */
int num_pixel;
int num_ray;
int num_thread;

int main(int argc, char** argv) {
	
	num_pixel = 1000;
	num_ray = 1e7;
	num_thread = 1;
	Camera cam;

	cam.pos = vec3(0.0, 10.0, 0.0);
	cam.height = 20.0;
	cam.width = 20.0;

	setArgs(argc, argv);

	srand(time(NULL));

	double dt = omp_get_wtime();
	rayTrace(num_pixel, num_pixel, num_ray, cam);
	dt = omp_get_wtime() - dt;

	printf("\n%d\t%d\t%lf\n", num_pixel*num_pixel, num_ray, dt);
}

void rayTrace(int px, int py, int nrays, Camera camera) {

	/** Basic variables */
	int n;

	/** Model parameters */
	double radius = 6.0;
	Vec3 vec_c = vec3(0.0, 12.0, 0.0); // position of sphere centre
	Vec3 vec_l = vec3(4.0, 4.0, -1.0); // position of light source

	/** Ray tracer variables global */
	double dotp_cc;

	/** Window display */
	double w_max_x, w_min_x, w_max_z, w_min_z;
	double * mat_grid; // vision grid

	/** Initialization */
	mat_grid = (double*) malloc(px * py * sizeof(double));
	memset(mat_grid, 0.0, px * py);
	w_max_x = camera.pos.x + camera.width * 0.5;
	w_min_x = camera.pos.x - camera.width * 0.5;
	w_max_z = camera.pos.z + camera.height * 0.5;
	w_min_z = camera.pos.z - camera.height * 0.5;
	dotp_cc = vec3DotP(vec_c, vec_c);

	omp_set_num_threads(num_thread);
	#pragma omp parallel for schedule(guided) shared(mat_grid) private(n)
	for (n=0; n<nrays; n++) {

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
			vec_v = randomDirection();
			vec_w = vec3Scale(vec_v, camera.pos.y / vec_v.y);
			dotp_vc = vec3DotP(vec_v, vec_c);
			delta = dotp_vc*dotp_vc + radius*radius - dotp_cc;
		} while (vec_w.x < w_min_x || vec_w.x > w_max_x || vec_w.z < w_min_z || vec_w.z > w_max_z || delta < 0);

		solution = dotp_vc - sqrt(delta);

		vec_i = vec3Scale(vec_v, solution);
		vec_n = vec3Combine(vec_i, vec_c, 1.0, -1.0);
		vec_n = vec3Scale(vec_n, 1.0 / sqrt(vec3DotP(vec_n, vec_n)));
		vec_s = vec3Combine(vec_l, vec_i, 1.0, -1.0);
		vec_s = vec3Scale(vec_s, 1.0 / sqrt(vec3DotP(vec_s, vec_s)));

		brightness = vec3DotP(vec_s, vec_n);
		brightness = MAX(brightness,0);

		j = px - 1 - (int) ((double) px * (vec_w.x - w_min_x) / (camera.width));
		i = (int) ((double) py * (vec_w.z - w_min_z) / (camera.height));
		
		/** To check progress */
		//if (n%(nrays/100)==0) {printf("."); fflush(stdout);}
		if (n%(nrays/100)==0) write(STDOUT_FILENO, ".", 2);

		#pragma omp atomic update
		mat_grid[i * px + j] += brightness;
	}

	save_grid(mat_grid, px, py, "output.cpu.out");

	free(mat_grid);
}

Vec3 randomDirection() {

	/*
	* return a unit vector of random direction.
	*/

	double angle_psi, angle_theta;
	Vec3 dest;

	angle_psi = rand_double(0, 2 * PI);
	angle_theta = rand_double(0, PI);

	dest.x = sin(angle_theta) * cos(angle_psi);
	dest.y = sin(angle_theta) * sin(angle_psi);
	dest.z = cos(angle_theta);

	return dest;
}

double rand_double(double start, double end) {

	/*
	* return a random double between start and end.
	*/

	double r = rand() / (double) RAND_MAX;
	double x = (start < end) ? (start) : (end);
	double y = (start < end) ? (end) : (start);
	return (y - x) * r + x;
}

void save_grid(double * data, int px, int py, char * fname) {

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
		{"thread", required_argument, 0, 't'},
		{0, 0, 0, 0}
	};

	/* Detect the end of the options. */
	while ( (ch = getopt_long(argc, argv, "n:r:t:", long_options, &option_index)) != -1 ) {
		switch (ch) {
			case 'n':
				num_pixel = atoi(optarg);
				break;
			case 'r':
				num_ray = atoi(optarg);
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
