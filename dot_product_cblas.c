#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>

int DEBUG = 0;

/* dnf install blas-devel */
/* gcc dot_product_cblas.c -I/usr/include/cblas -lcblas -o dot_product_cblas */

void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, float *A, int *ldA, float *B, int *ldB, float *beta , float *C, int *ldC);

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
   
   int N = 8, M = 8, L = 8;

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

   /* Run */
   cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, L, M, alpha, matrixA, lda, matrixB, ldb, beta, matrixC, ldc);

   end = clock();

   /* Result */
   if(DEBUG >= 1){
      printf("matrixC = matrixA * matrixB\n");
      matrix_print(matrixC, N, L);
   }

   printf("Time: %f sec\n", (float)(end - start) / CLOCKS_PER_SEC);

   free(matrixA);
   free(matrixB);
   free(matrixC);

}
