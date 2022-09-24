#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cublas_v2.h>

int DEBUG = 1;

/* nvcc dot_product_cublas.cu -lcublas -o dot_product_cublas */

void initialize_rand(float *matrix, int A, int B){

   for(int i = 0; i < B; i++){   /* column */
      for(int j= 0 ; j < A; j++){   /* row */
         matrix[j * B + i] = (float)rand() / RAND_MAX;
      }
   }

}

void initialize_zero(float *matrix, int A, int B){

   for(int i = 0; i < B; i++){   /* column */
      for(int j = 0 ; j < A; j++){  /* row */
         matrix[j * B + i] = 0;
      }
   }

}

void matrix_print(float *matrix, int A, int B){

   for(int i = 0; i < B; i++){   /* column */
      for(int j= 0 ; j < A; j++){   /* row */
          printf("%3f ", matrix[j * B + i]);
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

   int N = 16, M = 16, L = 16;

   float alpha = 1.0, beta = 0.0;
   int lda, ldb, ldc;

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

   lda = N;
   ldb = M;
   ldc = N;

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

   cublasHandle_t handle;
   cublasCreate(&handle);

   /* Run */
   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, L, M, &alpha, device_matrixA, lda, device_matrixB, ldb, &beta, device_matrixC, ldc);
   cudaMemcpy(host_matrixC, device_matrixC, sizeof(float) * N * L, cudaMemcpyDeviceToHost);

   cublasDestroy(handle);
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
