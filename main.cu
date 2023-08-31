#include "./camera.cuh"
#include "./hittable.cuh"
#include "./hittable_list.cuh"
#include "./material.cuh"
#include "./rtweekend.cuh"
#include "./sphere.cuh"
#include <ctime>
#include <curand_kernel.h>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__device__ inline float linear_to_gamma(float linear_component) {
  return sqrtf(linear_component);
}

struct colorint {
  int v[3];
  __device__ colorint() : v{0, 0, 0} {}
  __device__ colorint(int v0, int v1, int v2) : v{v0, v1, v2} {}
  __device__ colorint(const vec3 &color) {
    float r = linear_to_gamma(color.x());
    float g = linear_to_gamma(color.y());
    float b = linear_to_gamma(color.z());

    interval inte(0.000, 0.999);
    v[0] = static_cast<int>(256 * inte.clamp(r));
    v[1] = static_cast<int>(256 * inte.clamp(g));
    v[2] = static_cast<int>(256 * inte.clamp(b));
  }
  __host__ __device__ inline int x() const { return v[0]; }
  __host__ __device__ inline int y() const { return v[1]; }
  __host__ __device__ inline int z() const { return v[2]; }
};

__global__ void render(colorint *pixels, camera cam, curandState *state,
                       hittable_list **d_world);

__global__ void create_world(hittable **d_list, hittable_list **d_world,
                             material **d_mat);
__global__ void free_world(hittable **d_list, hittable_list **d_world,
                           material **d_mat, int hit_num, int mat_num);

__global__ void init_rand(curandState *state, unsigned int seed);

struct pic {
  char *content;
  int len;
  pic() {
    content = (char *)malloc((50 + 12 * 2560 * 1440) * sizeof(char));
    strcpy(content, "P3\n2560 1440\n255\n");
    len = strlen(content);
  }
  pic(int width, int height) {
    content = (char *)malloc((50 + 12 * width * height) * sizeof(char));
    sprintf(content, "P3\n%d %d\n255\n", width, height);
    len = strlen(content);
  }
  ~pic() { delete content; }

  void append(const colorint &v) {
    len += (sprintf(content + len, "%d %d %d\n", v.x(), v.y(), v.z()));
  }

  void save() {
    FILE *fptr = fopen("./rtw.ppm", "w");

    if (fptr == NULL) {
      printf("Failed to open file\n");
      exit(1);
    }

    fprintf(fptr, "%s", content);

    fclose(fptr);
  }
};

__constant__ int width;
__constant__ int height;

int main(int argc, char *argv[]) {

  float aspect_ratio = 16.0 / 9.0;
  int w, h;
  const int hit_num = 4;
  const int mat_num = 4;

  if (argc == 2) {
    w = atoi(argv[0]);
    h = atoi(argv[1]);
  } else {
    w = 2560;
    h = 1440;
  }
  cudaMemcpyToSymbol(width, &w, sizeof(int));
  cudaMemcpyToSymbol(height, &h, sizeof(int));
  pic picture(w, h);

  hittable **d_list;
  hittable_list **d_world;
  material **d_mat;
  cudaMalloc(&d_list, hit_num * sizeof(hittable *));
  cudaMalloc(&d_world, sizeof(hittable *));
  cudaMalloc(&d_mat, mat_num * sizeof(material *));
  create_world<<<1, 1>>>(d_list, d_world, d_mat);

  dim3 dimGrid((w + 15) / 16, (h + 15) / 16);
  dim3 dimBlock(16, 16);

  colorint *pixels;
  cudaMallocManaged(&pixels, w * h * sizeof(colorint));

  curandState *d_states;
  cudaMalloc(&d_states, w * h * sizeof(curandState));
  init_rand<<<dimGrid, dimBlock>>>(d_states, time(NULL));

  camera cam = camera(w, h);
  cudaDeviceSynchronize();

  render<<<dimGrid, dimBlock>>>(pixels, cam, d_states, d_world);
  cudaDeviceSynchronize();

  free_world<<<1, 1>>>(d_list, d_world, d_mat, hit_num, mat_num);

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      picture.append(pixels[y * w + x]);
    }
  }
  picture.save();
  cudaFree(pixels);
  return 0;
}

__global__ void create_world(hittable **d_list, hittable_list **d_world,
                             material **d_mat) {
  d_mat[0] = new lambertian(vec3(0.8, 0.8, 0));
  d_mat[1] = new lambertian(vec3(0.7, 0.3, 0.3));
  d_mat[2] = new metal(vec3(0.8, 0.8, 0.8));
  d_mat[3] = new metal(vec3(0.8, 0.6, 0.2));
  d_list[0] = new sphere(point3(0, -100.5, -1), 100, d_mat[0]);
  d_list[1] = new sphere(point3(0, 0, -1), 0.5, d_mat[1]);
  d_list[2] = new sphere(point3(-1, 0, -1), 0.5, d_mat[2]);
  d_list[3] = new sphere(point3(1, 0, -1), 0.5, d_mat[3]);
  d_world[0] = new hittable_list(d_list, 4);
}

__global__ void free_world(hittable **d_list, hittable_list **d_world,
                           material **d_mat, int hit_num, int mat_num) {
  for (int i = 0; i < hit_num; i++) {
    delete d_list[i];
  }
  for (int i = 0; i < mat_num; i++) {
    delete d_mat[i];
  }
  delete d_world[0];
}

__global__ void init_rand(curandState *state, unsigned int seed) {
  int id = (threadIdx.y + blockIdx.y * blockDim.y) * width + threadIdx.x +
           blockIdx.x * blockDim.x;
  curand_init(seed, id, 0, &state[id]);
}

__global__ void render(colorint *pixels, camera cam, curandState *states,
                       hittable_list **d_world) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int id = x + y * width;
  if (x >= cam.width || y >= cam.height)
    return;
  // auto pixel_center = pixel00_loc + (x * dpdu) + (y * dpdv);
  // auto ray_direction = pixel_center - camera_center;
  // ray r(camera_center, ray_direction);
  pixels[y * cam.width + x] =
      colorint(cam.render_ray(x, y, d_world, &states[id]));
}
