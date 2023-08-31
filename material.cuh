#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "./hittable.cuh"
#include "./rtweekend.cuh"

class material {
public:
  __device__ virtual ~material() {}
  __device__ virtual bool scatter(const ray &r_in, const hit_record &rec,
                                  vec3 &attenuation, ray &scattered,
                                  curandState *state) const = 0;
};

class lambertian : public material {
private:
  vec3 albedo;

public:
  __device__ lambertian(const vec3 &a) : albedo(a) {}
  __device__ bool scatter(const ray &r_in, const hit_record &rec,
                          vec3 &attenuation, ray &scattered,
                          curandState *state) const override {
    auto scatter_direction = rec.normal + random_unit(state);
    if (scatter_direction.near_zero())
      scatter_direction = rec.normal;
    scattered = ray(rec.p, scatter_direction);
    attenuation = albedo;
    return true;
  }
};

class metal : public material {
private:
  vec3 albedo;

public:
  __device__ metal(const vec3 &c) : albedo(c) {}
  __device__ bool scatter(const ray &r_in, const hit_record &rec,
                          vec3 &attenuation, ray &scattered,
                          curandState *state) const override {
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected);
    attenuation = albedo;
    return true;
  }
};

#endif
