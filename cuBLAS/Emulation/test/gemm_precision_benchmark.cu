/*
 * gemm_precision_benchmark.cu
 *
 * GEMM performance across four compute paths:
 *   1. INT8 TC        – cublasGemmEx  (CUDA_R_8I → CUDA_R_32I, Tensor Core)
 *   2. FP64-emu INT8  – cublasDgemm   (CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH, dynamic TC)
 *   3. FP16 TC        – cublasHgemm   (Tensor Core)
 *   4. FP64           – cublasDgemm   (CUDA cores, CUBLAS_DEFAULT_MATH)
 *
 * Sweep: m=n ∈ {16,64,256,1024,2048,4096} × k ∈ {16,64,256,1024,2048,4096}
 * Prints one TFLOP/s table per precision after all runs complete.
 *
 * Build:
 *   nvcc -O2 -arch=sm_90 gemm_precision_benchmark.cu \
 *        -I../../utils -lcublas -o gemm_precision_benchmark
 *   (sm_80 for A100, sm_86 for RTX 3090, sm_89 for RTX 4090, sm_100 for Blackwell)
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"

static const int MN_LIST[] = {16, 64, 256, 1024, 2048, 4096};
static const int K_LIST[]  = {16, 64, 256, 1024, 2048, 4096};
static const int NMN = 6, NK = 6;
static const int REPS    = 20;       // timed iterations (+ 1 warmup)
static const int MAX_DIM = 4096;
static const size_t MAX_ELEMS = (size_t)MAX_DIM * MAX_DIM;

/* 1 warmup + REPS timed calls; returns average ms */
template <typename F>
static double time_fn(cudaStream_t stream, F fn) {
    fn();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int r = 0; r < REPS; ++r) fn();
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return (double)ms / REPS;
}

/* Print NMN×NK table with row=m=n, col=k */
static void print_table(const char *title, double tf[NMN][NK]) {
    printf("\n=== %s (TFLOP/s) ===\n", title);
    printf("  m=n \\ k ");
    for (int j = 0; j < NK; ++j) printf("  %9d", K_LIST[j]);
    printf("\n  ---------");
    for (int j = 0; j < NK; ++j) printf("  ---------");
    printf("\n");
    for (int i = 0; i < NMN; ++i) {
        printf("  %7d ", MN_LIST[i]);
        for (int j = 0; j < NK; ++j) printf("  %9.4f", tf[i][j]);
        printf("\n");
    }
}

int main(void) {
    /* FP64-emu: dynamic mantissa, 79 bits = 10 INT8 slices */
    const int emMaxBits = 40;
    const cudaEmulationMantissaControl_t emCtrl = CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d)   Reps: %d (+1 warmup)\n\n",
           prop.name, prop.major, prop.minor, REPS);

    cublasHandle_t handle;
    cudaStream_t   stream;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    /* Pre-allocate emulation workspace for worst-case MAX_DIM^3 problem */
    void *workspace = nullptr;
    size_t wsSz = getFixedPointWorkspaceSizeInBytes(
        MAX_DIM, MAX_DIM, MAX_DIM, 1, false, emCtrl, emMaxBits);
    CUDA_CHECK(cudaMalloc(&workspace, wsSz));
    CUBLAS_CHECK(cublasSetWorkspace(handle, workspace, wsSz));
    printf("FP64-emu workspace: %.1f MB\n\n", wsSz / 1e6);

    /* Allocate device buffers once at MAX_DIM×MAX_DIM; reused for all sizes */
    int8_t  *d_Ai, *d_Bi; int32_t *d_Ci;
    __half  *d_Ah, *d_Bh, *d_Ch;
    double  *d_Ad, *d_Bd, *d_Cd;

    CUDA_CHECK(cudaMalloc(&d_Ai, MAX_ELEMS * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_Bi, MAX_ELEMS * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_Ci, MAX_ELEMS * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_Ah, MAX_ELEMS * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_Bh, MAX_ELEMS * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_Ch, MAX_ELEMS * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_Ad, MAX_ELEMS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Bd, MAX_ELEMS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Cd, MAX_ELEMS * sizeof(double)));

    /* Fill device buffers with random data (same data used for A and B per type) */
    {
        srand(42);
        std::vector<int8_t> hi(MAX_ELEMS);
        std::vector<__half> hh(MAX_ELEMS);
        std::vector<double> hd(MAX_ELEMS);
        for (size_t q = 0; q < MAX_ELEMS; ++q) {
            hi[q] = (int8_t)(rand() % 256 - 128);
            hh[q] = __float2half(2.f * rand() / RAND_MAX - 1.f);
            hd[q] = 2.0 * rand() / RAND_MAX - 1.0;
        }
        CUDA_CHECK(cudaMemcpy(d_Ai, hi.data(), MAX_ELEMS * sizeof(int8_t),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Bi, hi.data(), MAX_ELEMS * sizeof(int8_t),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Ah, hh.data(), MAX_ELEMS * sizeof(__half),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Bh, hh.data(), MAX_ELEMS * sizeof(__half),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Ad, hd.data(), MAX_ELEMS * sizeof(double),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Bd, hd.data(), MAX_ELEMS * sizeof(double),  cudaMemcpyHostToDevice));
    }

    double tf_i8[NMN][NK], tf_emu[NMN][NK], tf_fp16[NMN][NK], tf_fp64[NMN][NK];

    /* ------------------------------------------------------------------ */
    /* Pass 1: INT8 Tensor Core                                            */
    /* CUBLAS_COMPUTE_32I always routes through INT8 Tensor Cores.         */
    /* ------------------------------------------------------------------ */
    printf("Running INT8 TC ...\n");
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    for (int i = 0; i < NMN; ++i)
        for (int j = 0; j < NK; ++j) {
            const int m = MN_LIST[i], n = m, k = K_LIST[j];
            const int32_t alpha = 1, beta = 0;
            double ms = time_fn(stream, [&] {
                CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                    &alpha, d_Ai, CUDA_R_8I,  m,
                            d_Bi, CUDA_R_8I,  k,
                    &beta,  d_Ci, CUDA_R_32I, m,
                    CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
            });
            tf_i8[i][j] = 2.0 * m * n * k / (ms * 1e9);
            printf("  m=n=%-4d k=%-4d  %.4f TFLOP/s\n", m, k, tf_i8[i][j]);
            fflush(stdout);
        }

    /* ------------------------------------------------------------------ */
    /* Pass 2: FP64-emulation via INT8 Tensor Core (dynamic mantissa)     */
    /* ------------------------------------------------------------------ */
    printf("\nRunning FP64-emu INT8 TC ...\n");
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
    CUBLAS_CHECK(cublasSetEmulationStrategy(handle, CUBLAS_EMULATION_STRATEGY_EAGER));
    CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(handle, emCtrl));
    CUBLAS_CHECK(cublasSetFixedPointEmulationMaxMantissaBitCount(handle, emMaxBits));
    CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaBitOffset(handle, -8));
    for (int i = 0; i < NMN; ++i)
        for (int j = 0; j < NK; ++j) {
            const int m = MN_LIST[i], n = m, k = K_LIST[j];
            const double alpha = 1.0, beta = 0.0;
            double ms = time_fn(stream, [&] {
                CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                    &alpha, d_Ad, m, d_Bd, k, &beta, d_Cd, m));
            });
            tf_emu[i][j] = 2.0 * m * n * k / (ms * 1e9);
            printf("  m=n=%-4d k=%-4d  %.4f TFLOP/s\n", m, k, tf_emu[i][j]);
            fflush(stdout);
        }

    /* ------------------------------------------------------------------ */
    /* Pass 3: FP16 Tensor Core                                            */
    /* cublasHgemm uses FP16 TC by default on SM >= 7.0.                  */
    /* ------------------------------------------------------------------ */
    printf("\nRunning FP16 TC ...\n");
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    for (int i = 0; i < NMN; ++i)
        for (int j = 0; j < NK; ++j) {
            const int m = MN_LIST[i], n = m, k = K_LIST[j];
            const __half alpha = __float2half(1.f), beta = __float2half(0.f);
            double ms = time_fn(stream, [&] {
                CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                    &alpha, d_Ah, m, d_Bh, k, &beta, d_Ch, m));
            });
            tf_fp16[i][j] = 2.0 * m * n * k / (ms * 1e9);
            printf("  m=n=%-4d k=%-4d  %.4f TFLOP/s\n", m, k, tf_fp16[i][j]);
            fflush(stdout);
        }

    /* ------------------------------------------------------------------ */
    /* Pass 4: FP64 CUDA core (no Tensor Cores on Ampere/Ada for FP64)    */
    /* ------------------------------------------------------------------ */
    printf("\nRunning FP64 CUDA core ...\n");
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    for (int i = 0; i < NMN; ++i)
        for (int j = 0; j < NK; ++j) {
            const int m = MN_LIST[i], n = m, k = K_LIST[j];
            const double alpha = 1.0, beta = 0.0;
            double ms = time_fn(stream, [&] {
                CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                    &alpha, d_Ad, m, d_Bd, k, &beta, d_Cd, m));
            });
            tf_fp64[i][j] = 2.0 * m * n * k / (ms * 1e9);
            printf("  m=n=%-4d k=%-4d  %.4f TFLOP/s\n", m, k, tf_fp64[i][j]);
            fflush(stdout);
        }

    /* Summary tables */
    print_table("INT8 Tensor Core  (I8→I32)",    tf_i8);
    print_table("FP64-Emulation via INT8 TC",     tf_emu);
    print_table("FP16 Tensor Core",               tf_fp16);
    print_table("FP64 CUDA Core",                 tf_fp64);

    /* Cleanup */
    CUDA_CHECK(cudaFree(d_Ai)); CUDA_CHECK(cudaFree(d_Bi)); CUDA_CHECK(cudaFree(d_Ci));
    CUDA_CHECK(cudaFree(d_Ah)); CUDA_CHECK(cudaFree(d_Bh)); CUDA_CHECK(cudaFree(d_Ch));
    CUDA_CHECK(cudaFree(d_Ad)); CUDA_CHECK(cudaFree(d_Bd)); CUDA_CHECK(cudaFree(d_Cd));
    CUDA_CHECK(cudaFree(workspace));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
