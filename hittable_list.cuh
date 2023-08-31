#ifndef HITTABLE_LISST_CUH
#define HITTABLE_LISST_CUH

#include "./hittable.cuh"
#include "./interval.cuh"
#include <stdio.h>

class hittable_list:public hittable {
  public:
  hittable** objects;
  int len;
  __device__ hittable_list(){}
  __device__ hittable_list(hittable** l, int length):objects(l),len(length){}
  __device__ bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = ray_t.max;

    for (int i = 0; i < len; i++) {
      if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
      }
    }
    return hit_anything;
  }
};

#endif
