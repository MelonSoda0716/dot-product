#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include<omp.h>
#endif

int DEBUG = 0;

/* gcc dot_product.c -o dot_product */
/* gcc -fopenmp dot_product.c -o dot_product_mp */
/* pgcc dot_product.c -acc -ta=tesla -Minfo=accel -o dot_product_acc */

void dot_product(float *matrixA, float *matrixB, float *matrixC, int N, int M, int L){

   int i = 0;
   int j = 0;
   int k = 0;

   #ifdef _OPENMP
   if(DEBUG >= 1){
      printf("max threads %d\n", omp_get_max_threads());
   }
   #endif

   #pragma omp parallel for private(j, k)
   #pragma acc data copyin(matrixA[0:N*M]) copyin(matrixB[0:M*L]) copyout(matrixC[0:N*L])
   #pragma acc parallel
   #pragma acc loop independent
   for(i = 0; i < N; i++){
      #pragma acc loop independent
      for(j = 0; j < L; j++){
         #pragma acc loop seq
         for(k = 0; k < M; k++){
            matrixC[i * L + j] = matrixC[i * L + j] + (matrixA[i * M + k] * matrixB[k * L + j]);
            #ifdef _OPENMP
            if(DEBUG >= 3){
               printf("thread=%d mtrixC[%d](%f) = matrixC[%d] + (matrixA[%d](%f) * matrixB[%d](%f)) i=%d j=%d k=%d\n"
               , omp_get_thread_num(), i * L + j, matrixC[i * L + j], i * L + j, i * M + k, matrixA[i * M + k], k * L + j, matrixB[k * L + j], i ,j ,k);
            }
            #else
            if(DEBUG >= 3){
               printf("mtrixC[%d](%f) = matrixC[%d] + (matrixA[%d](%f) * matrixB[%d](%f)) i=%d j=%d k=%d\n"
               , i * L + j,matrixC[i * L + j], i * L + j, i * M + k, matrixA[i * M + k], k * L + j, matrixB[k * L + j], i ,j ,k);
            }
            #endif
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
      for(int j = 0 ; j < B; j++){   /* column */
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
   #ifdef _OPENMP
   double mp_start, mp_end;
   #endif

   int N = 16, M = 16, L = 16;

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

   float *matrixA;
   matrixA = (float *)malloc(sizeof(float) * N * M);

   float *matrixB;
   matrixB = (float *)malloc(sizeof(float) * M * L);

   float *matrixC;
   matrixC = (float *)malloc(sizeof(float) * N * L);

   srand(time(NULL));

   /* Initialize */
   initialize_rand(matrixA, N, M);
   initialize_rand(matrixB, M, L);
   initialize_zero(matrixC, N, L);

   /* Check */
   if(DEBUG >= 1){
      printf("matrixA\n");
      matrix_print(matrixA, N, M);
      printf("matrixB\n");
      matrix_print(matrixB, M, L);
   }

   start = clock();
   #ifdef _OPENMP
   mp_start = omp_get_wtime();
   #endif

   /* Run */
   dot_product(matrixA, matrixB, matrixC, N , M, L);

   #ifdef _OPENMP
   mp_end = omp_get_wtime();
   #endif   
   end = clock();

   /* Result */
   if(DEBUG >= 1){
      printf("matrixC = matrixA * matrixB\n");
      matrix_print(matrixC, N, L);
   }

   printf("Time: %f sec\n", (float)(end - start) / CLOCKS_PER_SEC);
   #ifdef _OPENMP
   printf("Time(MP): %lf sec\n", (mp_end - mp_start));
   #endif

   free(matrixA);
   free(matrixB);
   free(matrixC);

}
