#include "./rtweekend.cuh"
#include "./hittable.cuh"
#include "./sphere.cuh"
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct colorint {
  int v[3];
  __device__ colorint() : v{0, 0, 0} {}
  __device__ colorint(int v0, int v1, int v2) : v{v0, v1, v2} {}
  __device__ colorint(const vec3 &v)
      : v{static_cast<int>(255.999 * v[0]), static_cast<int>(255.999 * v[1]),
          static_cast<int>(255.999 * v[2])} {}
  __host__ __device__ inline int x() const { return v[0]; }
  __host__ __device__ inline int y() const { return v[1]; }
  __host__ __device__ inline int z() const { return v[2]; }
};

__global__ void render(colorint *pixels, vec3 pixel00_loc, point3 camera_ceenter, vec3 dpdu, vec3 dpdv, int width,
                       int height);

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

int main(int argc, char *argv[]) {

  int width;
  int height;
  float aspect_ratio = 16.0 / 9.0;

  if (argc == 2) {
    width = atoi(argv[0]);
    height = atoi(argv[1]);
  } else {
    width = 2560;
    height = 1440;
  }
  pic picture(width, height);
  float focal_length = 1.;
  float viewport_height = 2;
  float viewport_width = viewport_height * (static_cast<float>(width) / height);
  auto camera_center = point3(0, 0, 0);

  auto viewport_u = vec3(viewport_width, 0, 0);
  auto viewport_v = vec3(0, -viewport_height, 0);

  auto dpdu = viewport_u / width;
  auto dpdv = viewport_v / height;

  auto viewport_upper_left = camera_center - vec3(0, 0, focal_length) -
                             viewport_u / 2 - viewport_v / 2;
  auto pixel00_loc = viewport_upper_left + 0.5 * (dpdu + dpdv);

  dim3 dimGrid((width + 15) / 16, (height + 15) / 16);
  dim3 dimBlock(16, 16);

  colorint *pixels;
  cudaMallocManaged(&pixels, width * height * sizeof(colorint));

  render<<<dimGrid, dimBlock, 2 * sizeof(hittable)>>>(pixels, pixel00_loc, camera_center, dpdu, dpdv,
                                width, height);
  cudaDeviceSynchronize();

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      picture.append(pixels[y * width + x]);
    }
  }
  picture.save();
  cudaFree(pixels);
  return 0;
}

__device__ void create_world(sphere* world) {
  world[0] = sphere(point3(0., 0., -1.), 0.5);
  world[1] = sphere(point3(0., -100.5, -1.), 100.);
}

__device__ bool world_hit(sphere* world, int len, const ray &r, float ray_tmin, float ray_tmax, hit_record &rec) {
  hit_record temp_rec;
  bool hit_anything = false;
  float closest_so_far = ray_tmax;

  for (int i = 0; i < len; i++) {
    // printf("?");
    if (hit(&world[i], r, ray_tmin, closest_so_far, temp_rec)) {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      rec = temp_rec;
    }
  }
  return hit_anything;
}

__device__ colorint ray_color(const ray &r, sphere* world) {
  hit_record rec;
  if (world_hit(world, 2, r, 0., infinity, rec)) {
    // printf("normal: %f %f %f\n", rec.normal.x(), rec.normal.y(), rec.normal.z());
    return colorint(0.5 * (rec.normal + vec3(1., 1., 1.)));
  }
  vec3 unit_direction = unit_vector(r.direction());
  float a = 0.5 * (unit_direction.y() + 1.0);
  auto color = colorint((1. - a) * vec3(1., 1., 1.) + a * vec3(0.5, 0.7, 1.));
  return color;
}

__global__ void render(colorint *pixels, vec3 pixel00_loc, point3 camera_center,
                       vec3 dpdu, vec3 dpdv, int width, int height) {
  extern __shared__ char shared[];
  sphere* shared_sphere = (sphere*)shared;
  if (threadIdx.x == 0) create_world(shared_sphere);
  __syncthreads();
  
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  auto pixel_center = pixel00_loc + (x * dpdu) + (y * dpdv);
  auto ray_direction = pixel_center - camera_center;
  ray r(camera_center, ray_direction);
  pixels[y * width + x] = ray_color(r, shared_sphere);

}
