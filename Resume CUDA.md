- cudaMallocManaged() // Cuda Memory Allocation Managed
- __global__ // Specifier. Tells Cuda c++ compiler that this is a funtion  that runs on the GPU and can be called from CPU code.. These __global__ functions are known as kernels, and code that runs on the GPU is often called device code, while code that runs on the CPU is host code. Goes at the start of the code.
- cudaFree(x); // delete [] x;
- add<<<1, 1>>>(N, x, y); // this line launches one GPU thread to run add(). add on GPU Kernel. equivalent to : add(N, x, y); which adds on CPU KERNEL
- cudaDeviceSynchronize() // I need the CPU to wait until the kernel is done before it accesses the results (because CUDA kernel launches don’t block the calling CPU thread). To do this I just call cudaDeviceSynchronize() before doing the final error checking on the CPU.
- <<<1, 1>>> Execution configuration, and it tells the CUDA runtime how many parallel threads to use for the launch on the GPU. There are two parameters here, second one: the number of threads in a thread block.  CUDA GPUs run kernels using blocks of threads that are a multiple of 32 in size. If the second parameter is above one and you run the code with only this change, it will do the computation once per thread
- To spread the computation across the parallel threads you need to modify the kernel. CUDA C++ provides keywords that let kernels get the indices of the running threads:
    - threadIdx.x: contains the index of the current thread within its block
    - blockDim.x: contains the number of threads in the block. 
- modify the loop to stride through the array with parallel threads:
```c++
    __global__
    void add(int n, float *x, float *y)
    {
        int index = threadIdx.x;
        int stride = blockDim.x;
        for (int i = index; i < n; i += stride)
            y[i] = x[i] + y[i];
    }
```
- setting index to 0 and stride to 1 makes it semantically identical to the first version.
- CUDA GPUs have many parallel processors grouped into Streaming Multiprocessors, or SMs. Each SM can run multiple concurrent thread blocks. To take full advantage of the threads, one should launch the kernel with multiple thread blocks.
- First parameter of the execution configuration specifies the number of thread blocks. Together, the blocks of parallel threads make up what is known as the grid. We have N elements to process, and 256 threads per block so we need to calculate the number of blocks to get at least N threads. Simply divide N by the block size (being careful to round up in case N is not a multiple of blockSize). Ex:
    ``` c++
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);
    ```
    ![Figure 1](cuda_indexing.png)
- The figure above illustrates the approach to indexing into an array (one-dimensional) in CUDA using blockDim.x, gridDim.x, and threadIdx.x. The idea is that each thread gets its index by computing the offset to the beginning of its block (the block index times the block size: blockIdx.x * blockDim.x) and adding the thread’s index within the block (threadIdx.x). The code blockIdx.x * blockDim.x + threadIdx.x is idiomatic CUDA.
- gridDim.x: Contains the number of blocks in the grid
- blockIdx.x: Contains the index of the current thread block in the grid.  
    ```c++
    __global__
    void add(int n, float *x, float *y)
    {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
    }
    ```
    The above kernel sets stride to the total number of threads in the grid (blockDim.x * gridDim.x). This type of loop in a CUDA kernel is often called a grid-stride loop.

# Exercises

To keep you going, here are a few things to try on your own.

1. Browse the CUDA Toolkit documentation. If you haven’t installed CUDA yet, check out the Quick Start Guide and the installation guides. Then browse the Programming Guide and the Best Practices Guide. There are also tuning guides for various architectures.
2. Experiment with printf() inside the kernel. Try printing out the values of threadIdx.x and blockIdx.x for some or all of the threads. Do they print in sequential order? Why or why not?
3. Print the value of threadIdx.y or threadIdx.z (or blockIdx.y) in the kernel. (Likewise for blockDim and gridDim). Why do these exist? How do you get them to take on values other than 0 (1 for the dims)?
4. If you have access to a Pascal-based GPU, try running add_grid.cu on it. Is performance better or worse than the K80 results? Why? (Hint: read about Pascal’s Page Migration Engine and the CUDA 8 Unified Memory API.) For a detailed answer to this question, see the post Unified Memory for CUDA Beginners.

