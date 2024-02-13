//%%writefile add.cu //Genera archivo "add.cu"

#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20; // 1M elements

  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float)); //  float *x = new float[N];
  cudaMallocManaged(&y, N*sizeof(float)); //  float *y = new float[N];

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y); //  add(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize(); //only on Cuda. so the CPU won't call before is done

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x); //   delete [] x;
  cudaFree(y); //  delete [] y;
  
  return 0;
}
/*
%%shell //cmd or ctrl + j on VSCode
nvcc add.cu -o add_cuda// on c++: g++ add.cpp -o add // Compiles
./add_cuda// on c++: ./add //run

nvprof ./add_cuda // profile. find out how long the kernel takes to run
nvidia-smi // see the current GPU allocated to you
*/

