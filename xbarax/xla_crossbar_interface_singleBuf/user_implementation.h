// compile via:  gcc -shared -fPIC -o libfuncs.so xla_interface.c user_implementation.h
// when using c++ compiler make sure to use extern "C" to keep C++ name mangling from interfering

// extern "C" // to keep C++ name mangling from interfering if compiled with g++
void user_implementation_mul(const float* conductences, const float* vec_in, float* vec_out, unsigned int size_out, unsigned int size_in) {
  // example implementation of the matrix-vector multiplication
    // conductences: conductence values, stored in row-major order
    // vec_in: input vector
    // vec_out: output vector
    // size_out: number of rows of the conductences matrix
    // size_in: number of columns of the conductences matrix
  for (unsigned int i = 0; i < size_out; ++i) {
    vec_out[i] = 0.0;
    for (unsigned int j = 0; j < size_in; ++j) {
      vec_out[i] += conductences[i*size_in+j] * vec_in[j];
    }
  }
}

// extern "C" // to keep C++ name mangling from interfering if compiled with g++
void user_implementation_add(const float* conductences, const float* delta_conductences, float* conductences_out, unsigned int size) {
  // example implementation of the matrix-matrix addtion
    // conductences: conductence values, stored in row-major order
    // delta_conductences: values to add to conductances, stored in row-major order
    // conductences_out: result of the addition, stored in row-major order
    // size: number of elements of the conductences matrix
  for (unsigned int i = 0; i < size; ++i) {
    conductences_out[i] = conductences[i] + delta_conductences[i];
  }
}

#include <stdio.h> // just for printf

// extern "C" // to keep C++ name mangling from interfering if compiled with g++
void user_implementation_write(float *conductences, unsigned int size_out, unsigned int size_in){
//   const float *conductences = (float *)(conductences_void);
  // example implementation of the matrix-matrix addtion
    // conductences: conductence values, stored in row-major order
    // size_out: number of rows of the conductences matrix
    // size_in: number of columns of the conductences matrix
  printf("\nWRITING TO DEVICE, array of shape (%u x %u)\n", size_out, size_in);
  // for (unsigned int i = 0; i < size_out; ++i) {
  //   if (i > 0) {
  //     printf("\n [ ");
  //   } else {
  //     printf("[[ ");
  //   }  
  //   for (unsigned int j = 0; j < size_in; ++j) {
  //     printf("%f ", conductences[i*size_in+j]);
  //   }
  //   printf("]");
  // }
  // printf("]\n\n");
}