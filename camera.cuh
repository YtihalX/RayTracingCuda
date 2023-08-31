#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "./hittable.cuh"
#include "./material.cuh"
#include "./rtweekend.cuh"
#include "./vec3.cuh"
#include "hittable_list.cuh"
#include <curand_kernel.h>
#include <stdio.h>

class camera {
public:
  float aspect_ratio;
  int width;
  int height;
  point3 center;
  point3 pixel00_loc;
  vec3 dudp;
  vec3 dvdp;
  int dsamdp = 100;
  int max_depth = 50;

  camera(int w = 2560, int h = 1440) {
    width = w;
    height = h;
    aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
    height = (height < 1) ? 1 : height;

    center = point3(0, 0, 0);

    float focal_length = 1;
    float viewport_height = 2;
    float viewport_width = viewport_height * aspect_ratio;

    vec3 viewport_u = vec3(viewport_width, 0, 0);
    vec3 viewport_v = vec3(0, -viewport_height, 0);

    dudp = viewport_u / width;
    dvdp = viewport_v / height;

    auto viewport_upper_left =
        center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
    pixel00_loc = viewport_upper_left + 0.5 * (dudp + dvdp);
  }

  __device__ vec3 render_ray(int x, int y, hittable_list **d_world,
                             curandState *state) const {
    vec3 pixel_color(0, 0, 0);
    for (int sample = 0; sample < dsamdp; sample++) {
      auto pixel_center = pixel00_loc + (x * dudp) + (y * dvdp);
      auto pixel_sample = pixel_center + pixel_sample_disk(state);

      auto ray_origin = center;
      auto ray_direction = pixel_sample - center;

      pixel_color += ray_color(ray(ray_origin, ray_direction), d_world, state);
    }
    return pixel_color / dsamdp;
  }

private:
  __device__ vec3 ray_color(const ray &r, hittable_list **d_world,
                            curandState *state) const {
    ray recur_ray = r;
    vec3 color(1, 1, 1);
    int i;
    for (i = 0; i < max_depth; i++) {
      hit_record rec;
      if (d_world[0]->hit(recur_ray, interval(0.001, infinity), rec)) {
        vec3 attenuation;
        ray r_in = recur_ray;
        if (rec.mat->scatter(r_in, rec, attenuation, recur_ray, state)) {
          color = attenuation * color;
          continue;
        }
        color = vec3(0, 0, 0);
        break;
      }

      vec3 unit_direction = unit_vector(recur_ray.direction());
      float a = 0.5 * (unit_direction.y() + 1.);
      color = color * ((1 - a) * vec3(1, 1, 1) + a * vec3(0.5, 0.7, 1.));
      break;
    }
    if (i == max_depth)
      return vec3(0, 0, 0);
    return color;
  }

  __device__ vec3 pixel_sample_disk(curandState *state) const {
    float theta = curand_uniform(state) * 2 * pi;
    float r = curand_normal(state);
    return r * (dudp * cosf(theta) + dvdp * sinf(theta));
  }
};

#endif
