#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int DEBUG = 0;
int THREADS_PER_BLOCK = 8;

/* nvcc dot_product.cu -o dot_product_cu */

__global__ void dot_product(float *matrixA, float *matrixB, float *matrixC, int N, int M, int L, int DEBUG){
   
   int i = threadIdx.y + blockIdx.y * blockDim.y;
   int j = threadIdx.x + blockIdx.x * blockDim.x;
   int k = 0;
   
   if((i < N) && (j < L)){
      for(k = 0; k < M; k++){
         matrixC[i * L + j] = matrixC[i * L + j] + (matrixA[i * M + k] * matrixB[k * L + j]);
         if(DEBUG >= 3){
            printf("mtrixC[%d](%f) = matrixC[%d] + (matrixA[%d](%f) * matrixB[%d](%f)) i=%d j=%d k=%d\n"
            , i * L + j, matrixC[i * L + j], i * L + j, i * M + k, matrixA[i * M + k], k * L + j, matrixB[k * L + j], i ,j ,k);
         }
      }
   }

}

__global__ void dot_product_grid_stride_loops(float *matrixA, float *matrixB, float *matrixC, int N, int M, int L, int DEBUG){

   int i = threadIdx.y + blockIdx.y * blockDim.y;
   int j = threadIdx.x + blockIdx.x * blockDim.x;
   int k = 0;

   for(i = threadIdx.y + blockIdx.y * blockDim.y; i < N; i = i + blockDim.y * gridDim.y){
      for(j = threadIdx.x + blockIdx.x * blockDim.x; j < L; j = j + blockDim.x * gridDim.x){
         for(k = 0; k < M; k++){
            matrixC[i * L + j] = matrixC[i * L + j] + (matrixA[i * M + k] * matrixB[k * L + j]);
            if(DEBUG >= 3){
               printf("mtrixC[%d](%f) = matrixC[%d] + (matrixA[%d](%f) * matrixB[%d](%f)) i=%d j=%d k=%d\n"
               , i * L + j, matrixC[i * L + j], i * L + j, i * M + k, matrixA[i * M + k], k * L + j, matrixB[k * L + j], i ,j ,k);
            }
         }
      }
   }

}

void initialize_rand(float *matrix, int A, int B){

   for(int i = 0; i < A; i++){   /* row */
      for(int j= 0 ; j < B; j++){   /* column */
         matrix[i * B + j] = (float)rand() / RAND_MAX;
      }
   }

}

void initialize_zero(float *matrix, int A, int B){

   for(int i = 0; i < A; i++){   /* row */
      for(int j = 0 ; j < B; j++){  /* column */
         matrix[i * B + j] = 0;
      }
   }

}

void matrix_print(float *matrix, int A, int B){
   
   for(int i = 0; i < A; i++){   /* row */
      for(int j= 0 ; j < B; j++){   /* column */
          printf("%3f ", matrix[i * B + j]);
      }
      printf("\n");
   }
   printf("\n");
   
}

int main(int argc, char *argv[]){

   clock_t start, end;
   cudaEvent_t cu_start, cu_stop;
   float cuda_time = 0.0;

   cudaEventCreate(&cu_start);
   cudaEventCreate(&cu_stop);

   int N = 8, M = 8, L = 8;

   if(argc > 1){
      N = atoi(argv[1]);
   }
   
   if(argc > 2){
      M = atoi(argv[2]);
   }
   
   if(argc > 3){
      L = atoi(argv[3]);
   }

   if(argc > 4){
      DEBUG = atoi(argv[4]);
   }

   dim3 grid((L + THREADS_PER_BLOCK - 1 / THREADS_PER_BLOCK), (N + THREADS_PER_BLOCK - 1 / THREADS_PER_BLOCK));
   dim3 block(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

   float *host_matrixA, *device_matrixA;
   host_matrixA = (float *)malloc(sizeof(float) * N * M);
   cudaMalloc((void**)&device_matrixA, sizeof(float) * N * M );
   
   float *host_matrixB, *device_matrixB;
   host_matrixB = (float *)malloc(sizeof(float) * M * L);
   cudaMalloc((void**)&device_matrixB, sizeof(float) * M * L);

   float *host_matrixC, *device_matrixC;
   host_matrixC = (float *)malloc(sizeof(float) * N * L);
   cudaMalloc((void**)&device_matrixC, sizeof(float) * N * L);

   srand(time(NULL));
   
   /* Initialize */
   initialize_rand(host_matrixA, N, M);
   initialize_rand(host_matrixB, M, L);
   initialize_zero(host_matrixC, N, L);
 
   /* Check */
   if(DEBUG >= 1){
      printf("matrixA\n");
      matrix_print(host_matrixA, N, M);
      printf("matrixB\n");
      matrix_print(host_matrixB, M, L);
   }

   start = clock();
   cudaEventRecord(cu_start);

   cudaMemcpy(device_matrixA, host_matrixA, sizeof(float) * N * M, cudaMemcpyHostToDevice);
   cudaMemcpy(device_matrixB, host_matrixB, sizeof(float) * M * L, cudaMemcpyHostToDevice);
   cudaMemcpy(device_matrixC, host_matrixC, sizeof(float) * N * L, cudaMemcpyHostToDevice);

   /* Run */
   dot_product<<< grid, block >>>(device_matrixA, device_matrixB, device_matrixC, N, M, L, DEBUG);
   //dot_product_grid_stride_loops<<< grid, block >>>(device_matrixA, device_matrixB, device_matrixC, N, M, L, DEBUG);
   cudaDeviceSynchronize();

   cudaMemcpy(host_matrixA, device_matrixA, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
   cudaMemcpy(host_matrixB, device_matrixB, sizeof(float) * M * L, cudaMemcpyDeviceToHost);
   cudaMemcpy(host_matrixC, device_matrixC, sizeof(float) * N * L, cudaMemcpyDeviceToHost);

   cudaEventRecord(cu_stop);
   cudaEventSynchronize(cu_stop);
   cudaEventElapsedTime(&cuda_time, cu_start, cu_stop);
   end = clock(); 

   /* Result */
   if(DEBUG >= 1){
      printf("matrixC = matrixA * matrixB\n");
      matrix_print(host_matrixC, N, L);
   }

   printf("Time: %f sec\n", (float)(end - start) / CLOCKS_PER_SEC);
   printf("Time(CUDA): %f sec\n", (cuda_time / 1000));

   free(host_matrixA);
   free(host_matrixB);
   free(host_matrixC);
   cudaFree(device_matrixA);
   cudaFree(device_matrixB);
   cudaFree(device_matrixC);
   cudaEventDestroy(cu_start);
   cudaEventDestroy(cu_stop);

}
