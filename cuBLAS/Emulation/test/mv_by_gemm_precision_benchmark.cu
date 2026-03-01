/*
 * mv_by_gemm_precision_benchmark.cu
 *
 * Matrix-Vector multiply (y = A * x) performance across four compute paths,
 * implemented as GEMM with n=1:
 *   1. INT8 TC        – cublasGemmEx  (CUDA_R_8I → CUDA_R_32I, Tensor Core)
 *   2. FP64-emu INT8  – cublasDgemm   (CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH, dynamic TC)
 *   3. FP16 TC        – cublasHgemm   (Tensor Core)
 *   4. FP64           – cublasDgemm   (CUDA cores, CUBLAS_DEFAULT_MATH)
 *
 * Sweep: m (rows) ∈ {16,64,256,1024,2048,4096}
 *        k (cols) ∈ {1,16,64,256,1024,2048,4096}
 * Prints one GFLOP/s table and one GB/s table per precision.
 *
 * Build:
 *   nvcc -O2 -arch=sm_90 mv_by_gemm_precision_benchmark.cu \
 *        -I../../utils -lcublas -o mv_by_gemm_precision_benchmark
 *   (sm_80 for A100, sm_86 for RTX 3090, sm_89 for RTX 4090, sm_100 for Blackwell)
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"

static const int M_LIST[] = {16, 64, 256, 1024, 2048, 4096, 8192, 16384};
static const int K_LIST[]  = {4, 16, 64, 256, 1024, 2048, 4096, 8192, 16384};
static const int NM = 8, NK = 9;
static const int REPS    = 20;       // timed iterations (+ 1 warmup)
static const int MAX_M   = 16384;
static const int MAX_K   = 16384;
static const size_t MAX_A = (size_t)MAX_M * MAX_K;  // matrix elements
static const size_t MAX_V = (size_t)MAX_K;           // input vector length

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

/* Print NM×NK table with row=m, col=k */
static void print_table(const char *title, const char *unit, double val[NM][NK]) {
    printf("\n=== %s (%s) ===\n", title, unit);
    printf("      m \\ k ");
    for (int j = 0; j < NK; ++j) printf("  %9d", K_LIST[j]);
    printf("\n  ----------");
    for (int j = 0; j < NK; ++j) printf("  ---------");
    printf("\n");
    for (int i = 0; i < NM; ++i) {
        printf("  %8d  ", M_LIST[i]);
        for (int j = 0; j < NK; ++j) printf("  %9.4f", val[i][j]);
        printf("\n");
    }
}

int main(void) {
    /* FP64-emu: dynamic mantissa, 79 bits = 10 INT8 slices */
    const int emMaxBits = 79;
    const cudaEmulationMantissaControl_t emCtrl = CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d)   Reps: %d (+1 warmup)\n",
           prop.name, prop.major, prop.minor, REPS);
    printf("Operation: y = A * x  (GEMM with n=1)\n\n");

    cublasHandle_t handle;
    cudaStream_t   stream;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    /* Pre-allocate emulation workspace for worst-case MAX_M×1×MAX_K problem */
    void *workspace = nullptr;
    size_t wsSz = getFixedPointWorkspaceSizeInBytes(
        MAX_M, 1, MAX_K, 1, false, emCtrl, emMaxBits);
    CUDA_CHECK(cudaMalloc(&workspace, wsSz));
    CUBLAS_CHECK(cublasSetWorkspace(handle, workspace, wsSz));
    printf("FP64-emu workspace: %.1f MB\n\n", wsSz / 1e6);

    /* Device buffers: matrix A (MAX_M × MAX_K), input vector x (MAX_K),
       output vector y (MAX_M, INT32 for the INT8 path) */
    int8_t  *d_Ai, *d_xi; int32_t *d_yi;
    __half  *d_Ah, *d_xh, *d_yh;
    double  *d_Ad, *d_xd, *d_yd;

    CUDA_CHECK(cudaMalloc(&d_Ai, MAX_A * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_xi, MAX_V * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_yi, MAX_M * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_Ah, MAX_A * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_xh, MAX_V * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_yh, MAX_M * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_Ad, MAX_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_xd, MAX_V * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_yd, MAX_M * sizeof(double)));

    /* Fill with random data */
    {
        srand(42);
        std::vector<int8_t> hi_A(MAX_A), hi_x(MAX_V);
        std::vector<__half> hh_A(MAX_A), hh_x(MAX_V);
        std::vector<double> hd_A(MAX_A), hd_x(MAX_V);
        for (size_t q = 0; q < MAX_A; ++q) {
            hi_A[q] = (int8_t)(rand() % 256 - 128);
            hh_A[q] = __float2half(2.f * rand() / RAND_MAX - 1.f);
            hd_A[q] = 2.0 * rand() / RAND_MAX - 1.0;
        }
        for (size_t q = 0; q < MAX_V; ++q) {
            hi_x[q] = (int8_t)(rand() % 256 - 128);
            hh_x[q] = __float2half(2.f * rand() / RAND_MAX - 1.f);
            hd_x[q] = 2.0 * rand() / RAND_MAX - 1.0;
        }
        CUDA_CHECK(cudaMemcpy(d_Ai, hi_A.data(), MAX_A * sizeof(int8_t),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_xi, hi_x.data(), MAX_V * sizeof(int8_t),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Ah, hh_A.data(), MAX_A * sizeof(__half),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_xh, hh_x.data(), MAX_V * sizeof(__half),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Ad, hd_A.data(), MAX_A * sizeof(double),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_xd, hd_x.data(), MAX_V * sizeof(double),  cudaMemcpyHostToDevice));
    }

    double gf_i8  [NM][NK], gf_emu [NM][NK], gf_fp16[NM][NK], gf_fp64[NM][NK];
    double bw_i8  [NM][NK], bw_emu [NM][NK], bw_fp16[NM][NK], bw_fp64[NM][NK];

    /* ------------------------------------------------------------------ */
    /* Pass 1: INT8 TC   y(m×1) = A(m×k) * x(k×1)                        */
    /* CUBLAS_COMPUTE_32I always routes through INT8 Tensor Cores.         */
    /* ------------------------------------------------------------------ */
    printf("Running INT8 TC ...\n");
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    for (int i = 0; i < NM; ++i)
        for (int j = 0; j < NK; ++j) {
            const int m = M_LIST[i], k = K_LIST[j];
            const int32_t alpha = 1, beta = 0;
            double ms = time_fn(stream, [&] {
                CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, 1, k,
                    &alpha, d_Ai, CUDA_R_8I,  m,
                            d_xi, CUDA_R_8I,  k,
                    &beta,  d_yi, CUDA_R_32I, m,
                    CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
            });
            /* 2*m*k FLOPs; bytes: A(m*k*1) + x(k*1) + y(m*4) */
            gf_i8[i][j] = 2.0 * m * k / (ms * 1e6);
            bw_i8[i][j] = ((double)m * k + k + 4.0 * m) / (ms * 1e6);
            printf("  m=%-5d k=%-5d  %8.4f GFLOP/s  %8.3f GB/s\n",
                   m, k, gf_i8[i][j], bw_i8[i][j]);
            fflush(stdout);
        }

    /* ------------------------------------------------------------------ */
    /* Pass 2: FP64-emulation via INT8 TC (dynamic mantissa)              */
    /* ------------------------------------------------------------------ */
    printf("\nRunning FP64-emu INT8 TC ...\n");
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
    CUBLAS_CHECK(cublasSetEmulationStrategy(handle, CUBLAS_EMULATION_STRATEGY_EAGER));
    CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(handle, emCtrl));
    CUBLAS_CHECK(cublasSetFixedPointEmulationMaxMantissaBitCount(handle, emMaxBits));
    CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaBitOffset(handle, -8));
    for (int i = 0; i < NM; ++i)
        for (int j = 0; j < NK; ++j) {
            const int m = M_LIST[i], k = K_LIST[j];
            const double alpha = 1.0, beta = 0.0;
            double ms = time_fn(stream, [&] {
                CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, 1, k,
                    &alpha, d_Ad, m, d_xd, k, &beta, d_yd, m));
            });
            gf_emu[i][j] = 2.0 * m * k / (ms * 1e6);
            bw_emu[i][j] = 8.0 * ((double)m * k + k + m) / (ms * 1e6);
            printf("  m=%-5d k=%-5d  %8.4f GFLOP/s  %8.3f GB/s\n",
                   m, k, gf_emu[i][j], bw_emu[i][j]);
            fflush(stdout);
        }

    /* ------------------------------------------------------------------ */
    /* Pass 3: FP16 TC                                                     */
    /* cublasHgemm uses FP16 Tensor Cores by default on SM >= 7.0.        */
    /* ------------------------------------------------------------------ */
    printf("\nRunning FP16 TC ...\n");
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    for (int i = 0; i < NM; ++i)
        for (int j = 0; j < NK; ++j) {
            const int m = M_LIST[i], k = K_LIST[j];
            const __half alpha = __float2half(1.f), beta = __float2half(0.f);
            double ms = time_fn(stream, [&] {
                CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, 1, k,
                    &alpha, d_Ah, m, d_xh, k, &beta, d_yh, m));
            });
            gf_fp16[i][j] = 2.0 * m * k / (ms * 1e6);
            bw_fp16[i][j] = 2.0 * ((double)m * k + k + m) / (ms * 1e6);
            printf("  m=%-5d k=%-5d  %8.4f GFLOP/s  %8.3f GB/s\n",
                   m, k, gf_fp16[i][j], bw_fp16[i][j]);
            fflush(stdout);
        }

    /* ------------------------------------------------------------------ */
    /* Pass 4: FP64 CUDA cores (no Tensor Cores on Ampere/Ada for FP64)   */
    /* ------------------------------------------------------------------ */
    printf("\nRunning FP64 CUDA core ...\n");
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    for (int i = 0; i < NM; ++i)
        for (int j = 0; j < NK; ++j) {
            const int m = M_LIST[i], k = K_LIST[j];
            const double alpha = 1.0, beta = 0.0;
            double ms = time_fn(stream, [&] {
                CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, 1, k,
                    &alpha, d_Ad, m, d_xd, k, &beta, d_yd, m));
            });
            gf_fp64[i][j] = 2.0 * m * k / (ms * 1e6);
            bw_fp64[i][j] = 8.0 * ((double)m * k + k + m) / (ms * 1e6);
            printf("  m=%-5d k=%-5d  %8.4f GFLOP/s  %8.3f GB/s\n",
                   m, k, gf_fp64[i][j], bw_fp64[i][j]);
            fflush(stdout);
        }

    /* Summary tables — GFLOP/s */
    print_table("INT8 Tensor Core  (I8→I32)",  "GFLOP/s", gf_i8);
    print_table("FP64-Emulation via INT8 TC",  "GFLOP/s", gf_emu);
    print_table("FP16 Tensor Core",            "GFLOP/s", gf_fp16);
    print_table("FP64 CUDA Core",              "GFLOP/s", gf_fp64);

    /* Summary tables — effective memory bandwidth */
    print_table("INT8 Tensor Core  (I8→I32)",  "GB/s", bw_i8);
    print_table("FP64-Emulation via INT8 TC",  "GB/s", bw_emu);
    print_table("FP16 Tensor Core",            "GB/s", bw_fp16);
    print_table("FP64 CUDA Core",              "GB/s", bw_fp64);

    /* Cleanup */
    CUDA_CHECK(cudaFree(d_Ai)); CUDA_CHECK(cudaFree(d_xi)); CUDA_CHECK(cudaFree(d_yi));
    CUDA_CHECK(cudaFree(d_Ah)); CUDA_CHECK(cudaFree(d_xh)); CUDA_CHECK(cudaFree(d_yh));
    CUDA_CHECK(cudaFree(d_Ad)); CUDA_CHECK(cudaFree(d_xd)); CUDA_CHECK(cudaFree(d_yd));
    CUDA_CHECK(cudaFree(workspace));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
