#ifndef VEC3_CUH
#define VEC3_CUH

#include <curand_kernel.h>
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

  __host__ __device__ vec3 operator-() const {
    return vec3(-v[0], -v[1], -v[2]);
  }
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

  __host__ __device__ float modsq() const {
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  }

  __host__ __device__ float mod() const { return sqrt(modsq()); }

  __device__ bool near_zero() const {
    float s = 1e-8;
    return (fabs(v[0]) < s && fabs(v[1]) < s && fabs(v[2]) < s);
  }

  __device__ vec3 random(curandState *state) {
    return vec3(curand_normal(state), curand_normal(state),
                curand_normal(state));
  }
  __device__ vec3 random(float min, float max, curandState *state);
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

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
  return v * t;
}

__host__ __device__ inline vec3 operator/(const vec3 &v, float t) {
  return v * (1 / t);
}

__host__ __device__ inline float dot(const vec3 &u, const vec3 &v) {
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
  return vec3(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
              u[0] * v[1] - u[1] * v[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v) { return v / v.mod(); }

__device__ vec3 vec3::random(float min, float max, curandState *state) {
  return (max - min) * vec3(curand_normal(state), curand_normal(state),
                            curand_normal(state)) +
         vec3(min, min, min);
}

__device__ inline vec3 random_unit(curandState *state) {
  float pi = 3.14159265358979323846;
  float theta = curand_uniform(state) * pi;
  float phi = curand_uniform(state) * 2 * pi;
  return vec3(sinf(theta) * cosf(phi), sinf(theta) * sinf(phi), cosf(theta));
}

__device__ inline vec3 random_hemisphere(const vec3 &normal,
                                         curandState *state) {
  vec3 runit = random_unit(state);
  if (dot(runit, normal) > 0.) {
    return runit;
  }
  return -runit;
}

__device__ inline vec3 reflect(const vec3 &v, const vec3 &n) {
  return v - 2 * dot(v, n) * n;
}

#endif
