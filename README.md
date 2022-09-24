# dot-product
プログラミング練習用 内積(ドット積)

# Memo
MatrixC(NL) = MatrixA(NM) * MatrixB(ML)
```
./executable <N> <M> <L> <DEBUG>
```

# How to use
## GCC
Row-major order
```
$ gcc dot_product.c -o dot_product
$ ./dot_product 8 8 8 1
```

## GCC - OpenMP
Row-major order
```
$ gcc -fopenmp dot_product.c -o dot_product_mp
$ ./dot_product_mp 8 8 8 1
```

## GCC - BLAS
Column-major order
```
$ gcc dot_product_blas.c -lblas -o dot_product_blas
$ ./dot_product_blas 8 8 8 1
```

## GCC - CBLAS
Column-major order
```
$ gcc dot_product_cblas.c -I/usr/include/cblas -lcblas -o dot_product_cblas
$ ./dot_product_cblas 8 8 8 1
```

## CUDA
Row-major order  
Default: THREADS_PER_BLOCK = 8
```
$ nvcc dot_product.cu -o dot_product_cu
$ ./dot_product_cu 8 8 8 1
```

## CUDA - cuBLAS
Column-major order
```
$ nvcc dot_product_cublas.cu -lcublas -o dot_product_cublas
$ ./dot_product_cublas 8 8 8 1
```

## CUDA - cuBLASXt
Column-major order
```
$ nvcc dot_product_cublasxt.cu -lcublas -o dot_product_cublasxt
$ ./dot_product_cublasxt 8 8 8 1
```

## PGI - OpenACC
Row-major order  
Accelerator: NVIDIA GPU
```
$ pgcc dot_product.c -acc -ta=tesla -Minfo=accel -o dot_product_acc
$ ./dot_product_acc 8 8 8 1
```

## OpenMPI
Row-major order  
正方行列のみ  
N(行要素/列要素)をNP(プロセス数)で割り切れる必要がある
```
$ mpicc dot_product_mpi.c -o dot_product_mpi
$ mpiexec --hostfile hostfile -np 4 ./dot_product_mpi 8 8 8 1
```

