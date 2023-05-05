// gcc -shared -fPIC -o libfuncs.so xla_interface.c user_implementation.h

#include "user_implementation.h"

// template <typename T>
// extern "C" // to keep C++ name mangling from interfering if compiled with g++
void cpu_mcbmm_f32(void *out, const void **in) {
  const unsigned long *dims = (const unsigned long*)(in[0]);
  unsigned long dim0 = dims[0];
  unsigned long dim1 = dims[1];

  const float *conductences = (const float *)(in[1]);
  const float *b = (const float *)(in[2]);
  // const int *transpose_flag = (const int *)(in[3]);
  const int transpose_flag = 0;

  // if (transpose_flag){
  //   dim0 = dims[0];
  //   dim1 = dims[1];
  // } else {
  //   dim0 = dims[1];
  //   dim1 = dims[0];
  // }

  float *c = (float *)(out);

  user_implementation_mul(conductences, b, c, dim0, dim1);
}

// template <typename T>
// extern "C" // to keep C++ name mangling from interfering if compiled with g++
void cpu_mcbadd_f32(void *out, const void **in) {

  float *conductences_out = (float *)(out);

  const unsigned long *dims = (const unsigned long*)(in[0]);
  const unsigned long dim0 = dims[0];
  const unsigned long dim1 = dims[1];
  const unsigned long size = dim0 * dim1;

  const float *conductences = (const float *)(in[1]);
  const float *delta_conductences = (const float *)(in[2]);

  user_implementation_add(conductences, delta_conductences, conductences_out, size);
}
