#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "./hittable.cuh"
#include "./vec3.cuh"
#include "stdio.h"

class sphere : public hittable {
private:
  point3 center;
  float radius;

public:
  __device__ sphere(point3 center, float radius)
      : center(center), radius(radius) {}

  __device__ bool hit(const ray &r, interval ray_t,
                      hit_record &rec) const override {
    vec3 oc = r.origin() - center;
    float a = r.direction().modsq();
    float halfb = dot(oc, r.direction());
    float c = oc.modsq() - radius * radius;

    float discriminant = halfb * halfb - a * c;
    if (discriminant < 0)
      return false;
    float sqrtd = sqrtf(discriminant);

    float root = (-halfb - sqrtd) / a;
    if (!ray_t.surrounds(root)) {
      root = (-halfb + sqrtd) / a;
      if (!ray_t.surrounds(root))
        return false;
    }

    rec.t = root;
    rec.p = r.at(root);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);

    return true;
  }
  __device__ float radii() const { return radius; }
  __device__ point3 coc() const { return center; }
};

__device__ bool hit(const sphere *sp, const ray &r, float ray_tmin,
                    float ray_tmax, hit_record &rec) {
  vec3 oc = r.origin() - sp->coc();
  float a = r.direction().modsq();
  float halfb = dot(oc, r.direction());
  float c = oc.modsq() - sp->radii() * sp->radii();

  float discriminant = halfb * halfb - a * c;
  if (discriminant < 0)
    return false;
  float sqrtd = sqrtf(discriminant);

  float root = (-halfb - sqrtd) / a;
  if (root <= ray_tmin || ray_tmax <= root) {
    root = (-halfb + sqrtd) / a;
    if (root <= ray_tmin || ray_tmax <= root)
      return false;
  }

  rec.t = root;
  rec.p = r.at(root);
  vec3 outward_normal = (rec.p - sp->coc()) / sp->radii();
  rec.set_face_normal(r, outward_normal);
  // printf("%f\n", rec.normal.x());

  return true;
}

#endif
