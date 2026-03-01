/*
 * mv_precision_benchmark.cu
 *
 * Matrix-vector multiply  y = A * x  GFLOP/s across six compute paths:
 *
 *   Pass  API call                       Compute type / math mode          HW path
 *   ----  --------------------------------  --------------------------------  -------------------------
 *   1     cublasGemmEx (n=1)               CUBLAS_COMPUTE_32I                INT8  Tensor Core (≥SM7.5)
 *   2     cublasHgemm  (n=1)               DEFAULT_MATH                      FP16  Tensor Core (≥SM7.0)
 *   3     cublasGemmEx (n=1)               CUBLAS_COMPUTE_32F_FAST_TF32      FP32  TF32  TC     (≥SM8.0)
 *   4     cublasGemmEx (n=1)               CUBLAS_COMPUTE_16F_PEDANTIC       FP16  CUDA  cores
 *   5     cublasSgemv                      CUBLAS_PEDANTIC_MATH              FP32  CUDA  cores
 *   6     cublasDgemv                      DEFAULT_MATH                      FP64  CUDA  cores
 *
 * Notes on API choice:
 *   - cuBLAS provides cublasSgemv and cublasDgemv as dedicated Level-2 GEMV
 *     functions (used for passes 5 and 6).
 *   - No cublasHgemv exists; FP16 MV must go through cublasGemmEx/cublasHgemm.
 *   - CUBLAS_COMPUTE_16F_PEDANTIC disables Tensor Cores and keeps arithmetic
 *     in native FP16 on CUDA cores (pass 4).
 *   - CUBLAS_PEDANTIC_MATH on the handle disables TF32 substitution, keeping
 *     cublasSgemv on full FP32 CUDA cores (pass 5).
 *
 * Sweep:
 *   m ∈ {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}
 *   k ∈ {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}
 *   n = 1  (true MV, not GEMM)
 *
 * Metric: GFLOP/s = 2*m*k / (time_ms * 1e6)
 *
 * Output:
 *   stdout                   — per-pass GFLOP/s summary tables
 *   mv_precision_gflops.csv  — CSV for plot_mv_precision_heatmap.py
 *
 * Build (Linux):
 *   nvcc -O2 -std=c++11 -arch=sm_86 mv_precision_benchmark.cu \
 *        -I../../../utils -lcublas -o mv_precision_benchmark
 *
 *   sm_80=A100  sm_86=RTX3090/A5000  sm_89=RTX4090  sm_90=H100  sm_100=Blackwell
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"

/* ------------------------------------------------------------------ */
/* Sweep configuration                                                 */
/* ------------------------------------------------------------------ */
static const int M_LIST[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
static const int K_LIST[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
static const int NM = 9, NK = 9;
static const int REPS  = 50;
static const int MAX_M = 16384;
static const int MAX_K = 16384;

/* ------------------------------------------------------------------ */
/* Timing: 1 warmup + REPS timed calls; returns average ms            */
/* ------------------------------------------------------------------ */
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

/* ------------------------------------------------------------------ */
/* Print a NM × NK GFLOP/s table                                      */
/* ------------------------------------------------------------------ */
static void print_table(const char *title, double gf[NM][NK]) {
    printf("\n=== %s  (GFLOP/s) ===\n", title);
    printf("  m \\ k   ");
    for (int j = 0; j < NK; ++j) printf("  %8d", K_LIST[j]);
    printf("\n  ---------");
    for (int j = 0; j < NK; ++j) printf("  --------");
    printf("\n");
    for (int i = 0; i < NM; ++i) {
        printf("  %7d  ", M_LIST[i]);
        for (int j = 0; j < NK; ++j) printf("  %8.3f", gf[i][j]);
        printf("\n");
    }
}

/* ------------------------------------------------------------------ */
/* Write all six passes to CSV                                         */
/* ------------------------------------------------------------------ */
static void write_csv(const char *path,
                      double gf_i8_tc  [NM][NK],
                      double gf_fp16_tc[NM][NK],
                      double gf_fp32_tc[NM][NK],
                      double gf_fp16_cc[NM][NK],
                      double gf_fp32_cc[NM][NK],
                      double gf_fp64_cc[NM][NK]) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "WARNING: cannot write %s\n", path); return; }

    fprintf(f, "precision,m,k,gflops\n");

    static const char *names[6] = {
        "INT8_TC", "FP16_TC", "FP32_TC",
        "FP16_CC", "FP32_CC", "FP64_CC"
    };
    double (*tables[6])[NK] = {
        gf_i8_tc, gf_fp16_tc, gf_fp32_tc,
        gf_fp16_cc, gf_fp32_cc, gf_fp64_cc
    };

    for (int p = 0; p < 6; ++p)
        for (int i = 0; i < NM; ++i)
            for (int j = 0; j < NK; ++j)
                fprintf(f, "%s,%d,%d,%.4f\n",
                        names[p], M_LIST[i], K_LIST[j], tables[p][i][j]);

    fclose(f);
    printf("\nCSV written → %s\n", path);
}

/* ================================================================== */
/* main                                                                */
/* ================================================================== */
int main(void) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=================================================================\n");
    printf("  MV Precision Benchmark — y = A * x  (n=1)\n");
    printf("  Device : %s  (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  Reps   : %d timed  +  1 warmup  per (m, k) point\n", REPS);
    printf("  Passes : INT8 TC | FP16 TC | FP32 TC (TF32)\n");
    printf("           FP16 CC | FP32 CC (Sgemv) | FP64 CC (Dgemv)\n");
    printf("=================================================================\n\n");

    cublasHandle_t handle;
    cudaStream_t   stream;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    /* ----------------------------------------------------------------
     * Device buffers — allocated once at MAX_M × MAX_K, reused per (m,k).
     * The FP16 buffer is shared between the TC and CC passes.
     * The FP32 buffer is shared between the TC and CC passes.
     * ---------------------------------------------------------------- */
    const size_t szA = (size_t)MAX_M * MAX_K;
    const size_t szV = (size_t)MAX_K;
    const size_t szY = (size_t)MAX_M;

    int8_t  *d_Ai, *d_xi; int32_t *d_yi;
    __half  *d_Ah, *d_xh, *d_yh;
    float   *d_Af, *d_xf, *d_yf;
    double  *d_Ad, *d_xd, *d_yd;

    CUDA_CHECK(cudaMalloc(&d_Ai, szA * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_xi, szV * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_yi, szY * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_Ah, szA * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_xh, szV * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_yh, szY * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_Af, szA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_xf, szV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_yf, szY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ad, szA * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_xd, szV * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_yd, szY * sizeof(double)));

    {
        srand(42);
        std::vector<int8_t> hi_A(szA), hi_x(szV);
        std::vector<__half> hh_A(szA), hh_x(szV);
        std::vector<float>  hf_A(szA), hf_x(szV);
        std::vector<double> hd_A(szA), hd_x(szV);

        for (size_t q = 0; q < szA; ++q) {
            hi_A[q] = (int8_t)(rand() % 256 - 128);
            hh_A[q] = __float2half(2.f * rand() / RAND_MAX - 1.f);
            hf_A[q] = 2.f * rand() / RAND_MAX - 1.f;
            hd_A[q] = 2.0 * rand() / RAND_MAX - 1.0;
        }
        for (size_t q = 0; q < szV; ++q) {
            hi_x[q] = (int8_t)(rand() % 256 - 128);
            hh_x[q] = __float2half(2.f * rand() / RAND_MAX - 1.f);
            hf_x[q] = 2.f * rand() / RAND_MAX - 1.f;
            hd_x[q] = 2.0 * rand() / RAND_MAX - 1.0;
        }

        CUDA_CHECK(cudaMemcpy(d_Ai, hi_A.data(), szA*sizeof(int8_t),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_xi, hi_x.data(), szV*sizeof(int8_t),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Ah, hh_A.data(), szA*sizeof(__half),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_xh, hh_x.data(), szV*sizeof(__half),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Af, hf_A.data(), szA*sizeof(float),   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_xf, hf_x.data(), szV*sizeof(float),   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Ad, hd_A.data(), szA*sizeof(double),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_xd, hd_x.data(), szV*sizeof(double),  cudaMemcpyHostToDevice));
    }

    double gf_i8_tc  [NM][NK];
    double gf_fp16_tc[NM][NK];
    double gf_fp32_tc[NM][NK];
    double gf_fp16_cc[NM][NK];
    double gf_fp32_cc[NM][NK];
    double gf_fp64_cc[NM][NK];

    /* ==================================================================
     * Pass 1: INT8 Tensor Core
     *   CUBLAS_COMPUTE_32I routes through INT8 TC on SM >= 7.5.
     *   No Level-2 GEMV equivalent exists for INT8.
     * ================================================================== */
    printf("--- Pass 1/6: INT8 Tensor Core  [cublasGemmEx, COMPUTE_32I] ---\n");
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    for (int i = 0; i < NM; ++i)
        for (int j = 0; j < NK; ++j) {
            const int m = M_LIST[i], k = K_LIST[j];
            const int32_t alpha = 1, beta = 0;
            double ms = time_fn(stream, [&] {
                CUBLAS_CHECK(cublasGemmEx(
                    handle, CUBLAS_OP_N, CUBLAS_OP_N, m, 1, k,
                    &alpha, d_Ai, CUDA_R_8I,  m,
                            d_xi, CUDA_R_8I,  k,
                    &beta,  d_yi, CUDA_R_32I, m,
                    CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
            });
            gf_i8_tc[i][j] = 2.0 * m * k / (ms * 1e6);
            printf("  m=%-5d k=%-5d  %9.3f GFLOP/s\n", m, k, gf_i8_tc[i][j]);
            fflush(stdout);
        }

    /* ==================================================================
     * Pass 2: FP16 Tensor Core
     *   cublasHgemm uses FP16 TC by default on SM >= 7.0.
     *   No cublasHgemv exists in cuBLAS; GemmEx with n=1 is the only option.
     * ================================================================== */
    printf("\n--- Pass 2/6: FP16 Tensor Core  [cublasHgemm, n=1] ---\n");
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    for (int i = 0; i < NM; ++i)
        for (int j = 0; j < NK; ++j) {
            const int m = M_LIST[i], k = K_LIST[j];
            const __half alpha = __float2half(1.f), beta = __float2half(0.f);
            double ms = time_fn(stream, [&] {
                CUBLAS_CHECK(cublasHgemm(
                    handle, CUBLAS_OP_N, CUBLAS_OP_N, m, 1, k,
                    &alpha, d_Ah, m, d_xh, k, &beta, d_yh, m));
            });
            gf_fp16_tc[i][j] = 2.0 * m * k / (ms * 1e6);
            printf("  m=%-5d k=%-5d  %9.3f GFLOP/s\n", m, k, gf_fp16_tc[i][j]);
            fflush(stdout);
        }

    /* ==================================================================
     * Pass 3: FP32 Tensor Core (TF32)
     *   CUBLAS_COMPUTE_32F_FAST_TF32 routes FP32 GEMM through TF32 TC
     *   on SM >= 8.0.  No GEMV variant accepts a TF32 compute type.
     * ================================================================== */
    printf("\n--- Pass 3/6: FP32 Tensor Core (TF32)  [cublasGemmEx, COMPUTE_32F_FAST_TF32] ---\n");
    for (int i = 0; i < NM; ++i)
        for (int j = 0; j < NK; ++j) {
            const int m = M_LIST[i], k = K_LIST[j];
            const float alpha = 1.f, beta = 0.f;
            double ms = time_fn(stream, [&] {
                CUBLAS_CHECK(cublasGemmEx(
                    handle, CUBLAS_OP_N, CUBLAS_OP_N, m, 1, k,
                    &alpha, d_Af, CUDA_R_32F, m,
                            d_xf, CUDA_R_32F, k,
                    &beta,  d_yf, CUDA_R_32F, m,
                    CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT));
            });
            gf_fp32_tc[i][j] = 2.0 * m * k / (ms * 1e6);
            printf("  m=%-5d k=%-5d  %9.3f GFLOP/s\n", m, k, gf_fp32_tc[i][j]);
            fflush(stdout);
        }

    /* ==================================================================
     * Pass 4: FP16 CUDA cores
     *   CUBLAS_COMPUTE_16F_PEDANTIC disables Tensor Cores and computes
     *   in native FP16 on CUDA cores (2 FP16 ops per FP32 core per cycle
     *   on Ampere).  No cublasHgemv exists; GemmEx with n=1 is used.
     * ================================================================== */
    printf("\n--- Pass 4/6: FP16 CUDA cores  [cublasGemmEx, COMPUTE_16F_PEDANTIC] ---\n");
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    for (int i = 0; i < NM; ++i)
        for (int j = 0; j < NK; ++j) {
            const int m = M_LIST[i], k = K_LIST[j];
            const __half alpha = __float2half(1.f), beta = __float2half(0.f);
            double ms = time_fn(stream, [&] {
                CUBLAS_CHECK(cublasGemmEx(
                    handle, CUBLAS_OP_N, CUBLAS_OP_N, m, 1, k,
                    &alpha, d_Ah, CUDA_R_16F, m,
                            d_xh, CUDA_R_16F, k,
                    &beta,  d_yh, CUDA_R_16F, m,
                    CUBLAS_COMPUTE_16F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
            });
            gf_fp16_cc[i][j] = 2.0 * m * k / (ms * 1e6);
            printf("  m=%-5d k=%-5d  %9.3f GFLOP/s\n", m, k, gf_fp16_cc[i][j]);
            fflush(stdout);
        }

    /* ==================================================================
     * Pass 5: FP32 CUDA cores  — cublasSgemv (Level-2 GEMV)
     *   cublasSgemv is a dedicated matrix-vector function and does not
     *   accept a compute type argument.  CUBLAS_PEDANTIC_MATH on the
     *   handle prevents cuBLAS from substituting TF32 Tensor Cores,
     *   keeping the computation on full FP32 CUDA cores.
     * ================================================================== */
    printf("\n--- Pass 5/6: FP32 CUDA cores  [cublasSgemv, PEDANTIC_MATH] ---\n");
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    for (int i = 0; i < NM; ++i)
        for (int j = 0; j < NK; ++j) {
            const int m = M_LIST[i], k = K_LIST[j];
            const float alpha = 1.f, beta = 0.f;
            double ms = time_fn(stream, [&] {
                CUBLAS_CHECK(cublasSgemv(
                    handle, CUBLAS_OP_N, m, k,
                    &alpha, d_Af, m, d_xf, 1, &beta, d_yf, 1));
            });
            gf_fp32_cc[i][j] = 2.0 * m * k / (ms * 1e6);
            printf("  m=%-5d k=%-5d  %9.3f GFLOP/s\n", m, k, gf_fp32_cc[i][j]);
            fflush(stdout);
        }
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    /* ==================================================================
     * Pass 6: FP64 CUDA cores  — cublasDgemv (Level-2 GEMV)
     *   cublasDgemv is the dedicated double-precision MV function.
     *   FP64 TC exist only on data-center GPUs (A100, H100); on consumer
     *   GPUs (A5000, RTX series) this always uses CUDA cores.
     * ================================================================== */
    printf("\n--- Pass 6/6: FP64 CUDA cores  [cublasDgemv] ---\n");
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    for (int i = 0; i < NM; ++i)
        for (int j = 0; j < NK; ++j) {
            const int m = M_LIST[i], k = K_LIST[j];
            const double alpha = 1.0, beta = 0.0;
            double ms = time_fn(stream, [&] {
                CUBLAS_CHECK(cublasDgemv(
                    handle, CUBLAS_OP_N, m, k,
                    &alpha, d_Ad, m, d_xd, 1, &beta, d_yd, 1));
            });
            gf_fp64_cc[i][j] = 2.0 * m * k / (ms * 1e6);
            printf("  m=%-5d k=%-5d  %9.3f GFLOP/s\n", m, k, gf_fp64_cc[i][j]);
            fflush(stdout);
        }

    /* ---- Summary tables -------------------------------------------- */
    print_table("INT8  Tensor Core   [GemmEx, COMPUTE_32I]",          gf_i8_tc);
    print_table("FP16  Tensor Core   [Hgemm, n=1]",                   gf_fp16_tc);
    print_table("FP32  Tensor Core   [GemmEx, COMPUTE_32F_FAST_TF32]",gf_fp32_tc);
    print_table("FP16  CUDA cores    [GemmEx, COMPUTE_16F_PEDANTIC]",  gf_fp16_cc);
    print_table("FP32  CUDA cores    [Sgemv, PEDANTIC_MATH]",          gf_fp32_cc);
    print_table("FP64  CUDA cores    [Dgemv]",                         gf_fp64_cc);

    /* ---- Write CSV -------------------------------------------------- */
    write_csv("mv_precision_gflops.csv",
              gf_i8_tc, gf_fp16_tc, gf_fp32_tc,
              gf_fp16_cc, gf_fp32_cc, gf_fp64_cc);

    /* ---- Cleanup ---------------------------------------------------- */
    CUDA_CHECK(cudaFree(d_Ai)); CUDA_CHECK(cudaFree(d_xi)); CUDA_CHECK(cudaFree(d_yi));
    CUDA_CHECK(cudaFree(d_Ah)); CUDA_CHECK(cudaFree(d_xh)); CUDA_CHECK(cudaFree(d_yh));
    CUDA_CHECK(cudaFree(d_Af)); CUDA_CHECK(cudaFree(d_xf)); CUDA_CHECK(cudaFree(d_yf));
    CUDA_CHECK(cudaFree(d_Ad)); CUDA_CHECK(cudaFree(d_xd)); CUDA_CHECK(cudaFree(d_yd));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
