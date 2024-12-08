#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "files.h"
#include <nvtx3/nvToolsExt.h>

#define SOFTENING 1e-9f

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
 * Calculate the gravitational impact of all bodies in the system
 * on all others.
 */
__global__ void bodyForce(Body *p, float dt, int n) {
    // Calculate thread's global index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // Grid stride loop - each thread processes multiple bodies
    for (int i = tid; i < n; i += stride) {
        float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

        // Calculate forces from all other bodies
        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
        }

        // Update velocities
        p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
    }
}

__global__ void integratePosition(Body *p, float dt, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
    
}


int main(const int argc, const char** argv) {

  // The assessment will test against both 2<11 and 2<15.
  // Feel free to pass the command line argument 15 when you generate ./nbody report files
  int nBodies = 2<<11;
  if (argc > 1) nBodies = 2<<atoi(argv[1]);

  printf("bodies: %d \n", nBodies);

  // The assessment will pass hidden initialized values to check for correctness.
  // You should not make changes to these files, or else the assessment will not work.
  const char * initialized_values;
  const char * solution_values;

  int deviceId;
  cudaGetDevice(&deviceId);

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId); 

  // Calculate the number of threads per block and blocks per grid:
  int maxThreadsPerBlock = props.maxThreadsPerBlock;
   // Use the maximum threads per block supported by the GPU.
  int threadsPerBlock = maxThreadsPerBlock;   
  
  int numberOfBlocks = props.multiProcessorCount * 40; 

  if (nBodies == 2<<11) {
    initialized_values = "09-nbody/files/initialized_4096";
    solution_values = "09-nbody/files/solution_4096";
  } else { // nBodies == 2<<15
    initialized_values = "09-nbody/files/initialized_65536";
    solution_values = "09-nbody/files/solution_65536";
  }

  if (argc > 2) initialized_values = argv[2];
  if (argc > 3) solution_values = argv[3];

  const float dt = 0.01f; // Time step
  const int nIters = 10;  // Simulation iterations

  int bytes = nBodies * sizeof(Body);

  // Allocate host buffer for file I/O
  float *buf = (float *)malloc(bytes);
  Body *p = (Body*)buf;

  // Allocate unified memory
  Body *d_p;
  cudaMallocManaged(&d_p, bytes);

  // Read values from file into host buffer
  read_values_from_file(initialized_values, buf, bytes);

  // Copy initial values to unified memory
  memcpy(d_p, p, bytes);

  cudaMemPrefetchAsync(d_p, bytes, deviceId);
  
  double totalTime = 0.0;

  /*
   * This simulation will run for 10 cycles of time, calculating gravitational
   * interaction amongst bodies, and adjusting their positions to reflect.
   */

  for (int iter = 0; iter < nIters; iter++) {
    StartTimer();

  /*
   * You will likely wish to refactor the work being done in `bodyForce`,
   * and potentially the work to integrate the positions.
   */
    nvtxRangePush("Compute interbody forces");
    bodyForce<<<numberOfBlocks, threadsPerBlock>>>(d_p, dt, nBodies); // Launch the kernel 
    nvtxRangePop();

  /*
   * This position integration cannot occur until this round of `bodyForce` has completed.
   * Also, the next round of `bodyForce` cannot begin until the integration is complete.
   */
    nvtxRangePush("Integrate position");
    integratePosition<<<numberOfBlocks, threadsPerBlock>>>(d_p, dt, nBodies);  
    nvtxRangePop();
    const double tElapsed = GetTimer() / 1000.0;
    totalTime += tElapsed;
  }

  double avgTime = totalTime / (double)(nIters);
  float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;

  cudaDeviceSynchronize();
  cudaMemPrefetchAsync(d_p, bytes, cudaCpuDeviceId);

  // Copy final values back to host buffer for file writing
  memcpy(p, d_p, bytes);
  write_values_to_file(solution_values, buf, bytes);

  // You will likely enjoy watching this value grow as you accelerate the application,
  // but beware that a failure to correctly synchronize the device might result in
  // unrealistically high values.
  printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);
  printf("Total time: %f seconds\n", totalTime);

  // Cleanup
  free(buf);
  cudaFree(d_p);
}
