#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int DEBUG = 0;

/* mpicc dot_product_mpi.c -o dot_product_mpi */

void dot_product(float *matrixA, float *matrixB, float *matrixC, int N, int M, int L){

   int i = 0;
   int j = 0;
   int k = 0;
   int rank, size, hotname_length;
   char hostname[MPI_MAX_PROCESSOR_NAME];

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Get_processor_name(hostname, &hotname_length);

   for(i = 0; i < (N / size) && rank * (N / size) + i < N; i++){
      for(j = 0; j < L; j++){
         for(k = 0; k < M; k++){
            matrixC[i * L + j] = matrixC[i * L + j] + (matrixA[i * M + k] * matrixB[k * L + j]);
	         if(DEBUG >= 3){
               printf("%s rank=%d mtrixC[%d](%f) = matrixCS[%d] + (matrixA[%d](%f) * matrixB[%d](%f)) i=%d j=%d k=%d\n"
               , hostname, rank, i * L + j, matrixC[i * L + j], i * L + j, i * M + k, matrixA[i * M + k], k * L + j, matrixB[k * L + j], i ,j ,k);
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
   double mpi_start, mpi_end;

   int rank, size, hotname_length;
   char hostname[MPI_MAX_PROCESSOR_NAME];
   
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

   MPI_Init(&argc, &argv);

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Get_processor_name(hostname, &hotname_length);

   if(DEBUG >= 1){
      printf("%s: rank=%d size=%d\n", hostname, rank, size);
   }

   float *matrixA;
   matrixA = (float *)malloc(sizeof(float) * N * M);
   
   float *matrixB;
   matrixB = (float *)malloc(sizeof(float) * M * L);

   float *matrixC;
   matrixC = (float *)malloc(sizeof(float) * N * L);

   float *matrixAS;
   matrixAS = (float *)malloc(sizeof(float) * (N * M) / size);

   float *matrixCS;
   matrixCS = (float *)malloc(sizeof(float) * (N * L) / size);

   srand(time(NULL));
   
   /* Initialize */
   if(rank == 0){
      initialize_rand(matrixA, N, M);
      initialize_rand(matrixB, M, L);
      initialize_zero(matrixC, N, L);
      initialize_zero(matrixAS, 1, (N * M) / size);
      initialize_zero(matrixCS, 1, (N * L) / size);
   }
   else{
      initialize_zero(matrixAS, 1, (N * M) / size);
      initialize_zero(matrixB, M, L);
      initialize_zero(matrixCS, 1, (N * L) / size);
   }

   start = clock();
   mpi_start = MPI_Wtime();

   /* Send */
   MPI_Scatter(&matrixA[0], (N * M) / size, MPI_FLOAT, &matrixAS[0], (N * M) / size, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&matrixB[0], M * L, MPI_FLOAT, 0, MPI_COMM_WORLD);

   /* Check */
   if(DEBUG >= 1){
      if(rank == 0){
         printf("%s: matrixA\n", hostname);
         matrix_print(matrixA, N, M);
         printf("%s: matrixB\n", hostname);
         matrix_print(matrixB, M, L);
      }
      if(DEBUG >= 2){
         printf("%s: matrixAS\n", hostname);
         matrix_print(matrixAS, 1, (N * M) / size);
      }
   }

   /* Run */ 
   dot_product(matrixAS, matrixB, matrixCS, N, M, L);

   /* Receive */
   MPI_Gather(&matrixCS[0], (N * M) / size, MPI_FLOAT, &matrixC[0], (N * M) / size, MPI_FLOAT, 0, MPI_COMM_WORLD);
   
   mpi_end = MPI_Wtime();
   end = clock();

   /* Check */
   if(DEBUG >= 2){
      printf("%s: rank=%d matrixC = matrixA * matrixB\n", hostname, rank);
      matrix_print(matrixCS, 1, (N * L) / size);
   }

   /* Result */
   if(DEBUG >= 1){
      if(rank == 0){
         printf("%s: matrixC = matrixA * matrixB\n", hostname);
         matrix_print(matrixC, N, L);
      }
   }
   
   if(rank == 0){
      printf("Time: %f sec\n", (float)(end - start) / CLOCKS_PER_SEC);
      printf("Time(MPI): %lf sec\n", (mpi_end - mpi_start));
   }

   MPI_Finalize();
   free(matrixA);
   free(matrixB);
   free(matrixC);
   free(matrixAS);
   free(matrixCS);

}
