#ifndef VEC3_CUH
#define VEC3_CUH

#include <math.h>
#include <string.h>

class vec3 {
public:
  float v[3];

  __host__ __device__ vec3() : v{0, 0, 0} {}
  __host__ __device__ vec3(float v0, float v1, float v2) : v{v0, v1, v2} {}

  __host__ __device__ float x() const { return v[0]; }
  __host__ __device__ float y() const { return v[1]; }
  __host__ __device__ float z() const { return v[2]; }

  __host__ __device__ vec3 operator-() const { return vec3(-v[0], -v[1], -v[2]); }
  __host__ __device__ float operator[](int i) const { return v[i]; }
  __host__ __device__ float &operator[](int i) { return v[i]; }

  __host__ __device__ vec3 &operator+=(const vec3 &vec) {
    v[0] += vec[0];
    v[1] += vec[1];
    v[2] += vec[2];
    return *this;
  }

  __host__ __device__ vec3 &operator*=(float t) {
    v[0] *= t;
    v[1] *= t;
    v[2] *= t;
    return *this;
  }

  __host__ __device__ vec3 &operator/=(float t) {
    v[0] /= t;
    v[1] /= t;
    v[2] /= t;
    return *this;
  }

  __host__ __device__ float modsq() const { return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]; }

  __host__ __device__ float mod() const { return sqrt(modsq()); }
};

using point3 = vec3;

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v) {
  return vec3(u[0] + v[0], u[1] + v[1], u[2] + v[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v) {
  return vec3(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v) {
  return vec3(u[0] * v[0], u[1] * v[1], u[2] * v[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
  return vec3(t * v[0], t * v[1], t * v[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) { return v * t; }

__host__ __device__ inline vec3 operator/(const vec3 &v, float t) { return v * (1 / t); }

__host__ __device__ inline float dot(const vec3 &u, const vec3 &v) {
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
  return vec3(u[1] * v[2] - u[2] * v[1], 
              u[2] * v[0] - u[0] * v[2],
              u[0] * v[1] - u[1] * v[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
  return v / v.mod();
}

#endif
