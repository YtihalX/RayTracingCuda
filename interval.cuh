#ifndef INTERVAL_CUH
#define INTERVAL_CUH

#include "./rtweekend.cuh"

class interval {
  public:
  float min, max;
  __host__ __device__ interval():min(+infinity), max(-infinity) {}
  __host__ __device__ interval(float _min, float _max):min(_min), max(_max) {}
  __device__ bool contains(float x) const {
    return min <= x && x <= max;
  }
  __device__ bool surrounds(float x) const {
    return  min < x && x < max;
  }
  __device__ float clamp(float x) const {
    if (x < min) return min;
    if (x > max) return max;
    return x;
  }
  static const interval empty, universe;
};

const static interval empty(+infinity, -infinity);
const static interval universe(-infinity, +infinity);

#endif
