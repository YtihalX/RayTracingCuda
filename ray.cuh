#ifndef RAY_CUH
#define RAY_CUH

#include "vec3.cuh"

class ray {
private:
  point3 orig;
  vec3 dir;

public:
  __device__ ray() {}

  __device__ ray(const point3 &origin, const vec3 &direction)
  __device__     : orig(origin), dir(direction) {}

  __device__ point3 origin() const { return orig; }
  __device__ vec3 direction() const { return dir; }

  __device__ point3 at(float t) const { return orig + t * dir; }
};

#endif
