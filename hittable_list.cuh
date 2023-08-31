#ifndef HITTABLE_LISST_CUH
#define HITTABLE_LISST_CUH

#include "./hittable.cuh"

class hittable_list:public hittable {
  public:
  hittable** objects;
  int len;
  __device__ hittable_list(){}
  __device__ hittable_list(hittable** l, int length):objects(l),len(length){}
  __device__ bool hit(const ray &r, float ray_tmin, float ray_tmax, hit_record &rec) const override {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = ray_tmax;

    for (int i = 0; i < len; i++) {
      if (objects[i]->hit(r, ray_tmin, closest_so_far, temp_rec)) {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
      }
    }
    return hit_anything;
  }
};

#endif
