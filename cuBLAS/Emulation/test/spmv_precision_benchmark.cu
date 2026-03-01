/*
 * spmv_precision_benchmark.cu
 *
 * Sparse Matrix-Vector multiply (y = A * x) performance across four compute paths
 * using the cuSPARSE generic SpMV API (CSR format, random sparse matrices):
 *
 *   1. INT8      – CUDA_R_8I  matrix/vector → CUDA_R_32I  output
 *                  (integer arithmetic; integer SpMV)
 *   2. FP16 TC   – CUDA_R_16F matrix/vector → CUDA_R_16F  output, CUDA_R_32F compute
 *                  (CUSPARSE_SPMV_CSR_ALG2 → Tensor Core path on SM >= 7.0)
 *   3. FP32      – CUDA_R_32F matrix/vector → CUDA_R_32F  output
 *                  (TF32 Tensor Core may be used internally on Ampere+)
 *   4. FP64      – CUDA_R_64F matrix/vector → CUDA_R_64F  output (CUDA cores)
 *
 * Sweep:
 *   m = k ∈ {16, 64, 256, 1024, 2048, 4096}   (square matrix)
 *   density ∈ {0.1, 0.01, 0.001}               (fraction of nonzeros; Bernoulli sampling)
 *
 * Reports GFLOP/s and effective memory bandwidth (GB/s) for each combination.
 * GFLOP/s = 2 * nnz / time  (one multiply + one add per nonzero).
 * Bandwidth counts CSR structure (rowPtr, colInd, values) + input/output vectors.
 *
 * Build:
 *   nvcc -O2 -arch=sm_90 spmv_precision_benchmark.cu \
 *        -lcusparse -o spmv_precision_benchmark
 *   (sm_80 for A100, sm_86 for RTX 3090, sm_89 for RTX 4090, sm_100 for Blackwell)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse.h>

/* ---- Error-check macros ---- */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(_e), __FILE__, __LINE__);                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUSPARSE_CHECK(call)                                                    \
    do {                                                                        \
        cusparseStatus_t _s = (call);                                           \
        if (_s != CUSPARSE_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuSPARSE error %d at %s:%d\n",                    \
                    (int)_s, __FILE__, __LINE__);                               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* ---- Benchmark parameters ---- */
static const int    SIZE_LIST[]    = {16, 64, 256, 1024, 2048, 4096};
static const double DENSITY_LIST[] = {0.1, 0.01, 0.001};
static const int NSIZES   = (int)(sizeof(SIZE_LIST)    / sizeof(SIZE_LIST[0]));
static const int NDENSITY = (int)(sizeof(DENSITY_LIST) / sizeof(DENSITY_LIST[0]));
static const int REPS     = 20;   /* timed iterations per config (+1 warmup) */

enum PrecMode { I8 = 0, F16 = 1, F32 = 2, F64 = 3, NPREC = 4 };

static const char *PREC_LABEL[NPREC] = {
    "INT8  (I8->I32, integer)",
    "FP16  (F16->F16, cmp F32, CSR_ALG2 TC)",
    "FP32  (F32->F32, TF32 TC on Ampere+)",
    "FP64  (F64->F64, CUDA cores)",
};

/* ---- Timer: 1 warmup + REPS timed calls; returns average ms ---- */
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

/* ---- Build random CSR (Bernoulli sampling) on host ---- */
struct HostCSR {
    std::vector<int> rowPtr;  /* length m+1 */
    std::vector<int> colInd;  /* length nnz */
    int nnz;
};

static HostCSR buildRandomCSR(int m, int k, double density, unsigned seed) {
    HostCSR csr;
    csr.rowPtr.reserve(m + 1);
    csr.rowPtr.push_back(0);

    std::mt19937 rng(seed);
    std::bernoulli_distribution bern(density);

    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < k; ++c)
            if (bern(rng)) csr.colInd.push_back(c);
        csr.rowPtr.push_back((int)csr.colInd.size());
    }
    csr.nnz = (int)csr.colInd.size();
    return csr;
}

/* ---- Print one summary table (NSIZES × NDENSITY) ---- */
static void print_table(const char *title, const char *unit,
                        double val[NSIZES][NDENSITY]) {
    printf("\n=== %s  (%s) ===\n", title, unit);
    printf("   m=k  \\ density");
    for (int di = 0; di < NDENSITY; ++di)
        printf("  %10.3f", DENSITY_LIST[di]);
    printf("\n  --------");
    for (int di = 0; di < NDENSITY; ++di) printf("  ----------");
    printf("\n");
    for (int si = 0; si < NSIZES; ++si) {
        printf("  %6d  ", SIZE_LIST[si]);
        for (int di = 0; di < NDENSITY; ++di)
            printf("  %10.4f", val[si][di]);
        printf("\n");
    }
}

/* ==================================================================== */
int main(void) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device  : %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Kernel  : y = A * x  (CSR, Bernoulli random sparse matrix)\n");
    printf("Reps    : %d  (+1 warmup)\n\n", REPS);

    cusparseHandle_t handle;
    cudaStream_t     stream;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSPARSE_CHECK(cusparseSetStream(handle, stream));

    /* Results indexed as [precision][size_idx][density_idx] */
    double gf[NPREC][NSIZES][NDENSITY];
    double bw[NPREC][NSIZES][NDENSITY];
    memset(gf, 0, sizeof(gf));
    memset(bw, 0, sizeof(bw));

    /* ============================================================
     * Outer sweep: matrix size × density
     * For each (size, density): build one CSR topology, then run
     * all four precision modes sharing the same sparsity pattern.
     * ============================================================ */
    for (int si = 0; si < NSIZES; ++si) {
        const int m = SIZE_LIST[si];
        const int k = m;   /* square matrix */

        for (int di = 0; di < NDENSITY; ++di) {
            const double density = DENSITY_LIST[di];

            /* Shared sparsity structure */
            HostCSR csr = buildRandomCSR(m, k, density,
                                         /*seed=*/42 + si * 100 + di);
            const int nnz = csr.nnz;

            printf("m=k=%-5d  density=%.3f  nnz=%-8d\n", m, density, nnz);

            if (nnz == 0) {
                printf("  (skipped: no nonzeros)\n\n");
                continue;
            }

            /* Generate value arrays for each precision using the same seed */
            std::vector<int8_t> vals_i8(nnz),  x_i8(k);
            std::vector<__half> vals_f16(nnz), x_f16(k);
            std::vector<float>  vals_f32(nnz), x_f32(k);
            std::vector<double> vals_f64(nnz), x_f64(k);
            {
                std::mt19937 rng(9999);
                std::uniform_real_distribution<float> dist(-1.f, 1.f);
                for (int q = 0; q < nnz; ++q) {
                    float v = dist(rng);
                    vals_i8[q]  = (int8_t)(v * 127.f);
                    vals_f16[q] = __float2half(v);
                    vals_f32[q] = v;
                    vals_f64[q] = (double)v;
                }
                for (int q = 0; q < k; ++q) {
                    float v = dist(rng);
                    x_i8[q]  = (int8_t)(v * 127.f);
                    x_f16[q] = __float2half(v);
                    x_f32[q] = v;
                    x_f64[q] = (double)v;
                }
            }

            /* Shared device buffers for CSR indices (reused for all precisions) */
            int *d_rowPtr = nullptr, *d_colInd = nullptr;
            CUDA_CHECK(cudaMalloc(&d_rowPtr, (m + 1) * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_colInd,  nnz    * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_rowPtr, csr.rowPtr.data(),
                                  (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_colInd, csr.colInd.data(),
                                   nnz    * sizeof(int), cudaMemcpyHostToDevice));

            /* ---- Mode 0: INT8  (I8 → I32) ---- */
            {
                int8_t  *d_vals, *d_x;
                int32_t *d_y;
                CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(int8_t)));
                CUDA_CHECK(cudaMalloc(&d_x,    k   * sizeof(int8_t)));
                CUDA_CHECK(cudaMalloc(&d_y,    m   * sizeof(int32_t)));
                CUDA_CHECK(cudaMemcpy(d_vals, vals_i8.data(),
                                      nnz * sizeof(int8_t), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_x, x_i8.data(),
                                      k * sizeof(int8_t), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(d_y, 0, m * sizeof(int32_t)));

                cusparseSpMatDescr_t matA;
                cusparseDnVecDescr_t vecX, vecY;
                CUSPARSE_CHECK(cusparseCreateCsr(&matA, m, k, nnz,
                    d_rowPtr, d_colInd, d_vals,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_8I));
                CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, k, d_x, CUDA_R_8I));
                CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, m, d_y, CUDA_R_32I));

                const int32_t alpha = 1, beta = 0;
                size_t bufSz = 0;
                CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                    handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, vecX, &beta, vecY,
                    CUDA_R_32I, CUSPARSE_SPMV_ALG_DEFAULT, &bufSz));
                void *buf = nullptr;
                if (bufSz > 0) CUDA_CHECK(cudaMalloc(&buf, bufSz));

                double ms = time_fn(stream, [&] {
                    CUSPARSE_CHECK(cusparseSpMV(
                        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, vecX, &beta, vecY,
                        CUDA_R_32I, CUSPARSE_SPMV_ALG_DEFAULT, buf));
                });

                gf[I8][si][di] = 2.0 * nnz / (ms * 1e6);
                /* Bytes: vals(1B) + colInd(4B) + rowPtr(4B) + x(1B) + y(4B) */
                double bytes = (double)nnz   * (1 + 4)     /* vals + colInd */
                             + (double)(m+1) *  4           /* rowPtr        */
                             + (double)k     *  1           /* x int8        */
                             + (double)m     *  4;          /* y int32       */
                bw[I8][si][di] = bytes / (ms * 1e6);

                printf("  INT8 : %8.4f GFLOP/s  %8.3f GB/s\n",
                       gf[I8][si][di], bw[I8][si][di]);
                fflush(stdout);

                if (buf) CUDA_CHECK(cudaFree(buf));
                CUSPARSE_CHECK(cusparseDestroySpMat(matA));
                CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
                CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
                CUDA_CHECK(cudaFree(d_vals));
                CUDA_CHECK(cudaFree(d_x));
                CUDA_CHECK(cudaFree(d_y));
            }

            /* ---- Mode 1: FP16 TC  (F16 → F16, compute F32) ---- */
            /* CUSPARSE_SPMV_CSR_ALG2 routes through HMMA Tensor Cores on SM>=7.0 */
            {
                __half *d_vals, *d_x, *d_y;
                CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(__half)));
                CUDA_CHECK(cudaMalloc(&d_x,    k   * sizeof(__half)));
                CUDA_CHECK(cudaMalloc(&d_y,    m   * sizeof(__half)));
                CUDA_CHECK(cudaMemcpy(d_vals, vals_f16.data(),
                                      nnz * sizeof(__half), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_x, x_f16.data(),
                                      k * sizeof(__half), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(d_y, 0, m * sizeof(__half)));

                cusparseSpMatDescr_t matA;
                cusparseDnVecDescr_t vecX, vecY;
                CUSPARSE_CHECK(cusparseCreateCsr(&matA, m, k, nnz,
                    d_rowPtr, d_colInd, d_vals,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F));
                CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, k, d_x, CUDA_R_16F));
                CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, m, d_y, CUDA_R_16F));

                const float alpha = 1.0f, beta = 0.0f;
                const cusparseSpMVAlg_t alg = CUSPARSE_SPMV_CSR_ALG2;
                size_t bufSz = 0;
                CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                    handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, vecX, &beta, vecY,
                    CUDA_R_32F, alg, &bufSz));
                void *buf = nullptr;
                if (bufSz > 0) CUDA_CHECK(cudaMalloc(&buf, bufSz));

                double ms = time_fn(stream, [&] {
                    CUSPARSE_CHECK(cusparseSpMV(
                        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, vecX, &beta, vecY,
                        CUDA_R_32F, alg, buf));
                });

                gf[F16][si][di] = 2.0 * nnz / (ms * 1e6);
                double bytes = (double)nnz   * (2 + 4)     /* vals + colInd */
                             + (double)(m+1) *  4           /* rowPtr        */
                             + (double)k     *  2           /* x fp16        */
                             + (double)m     *  2;          /* y fp16        */
                bw[F16][si][di] = bytes / (ms * 1e6);

                printf("  FP16 : %8.4f GFLOP/s  %8.3f GB/s\n",
                       gf[F16][si][di], bw[F16][si][di]);
                fflush(stdout);

                if (buf) CUDA_CHECK(cudaFree(buf));
                CUSPARSE_CHECK(cusparseDestroySpMat(matA));
                CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
                CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
                CUDA_CHECK(cudaFree(d_vals));
                CUDA_CHECK(cudaFree(d_x));
                CUDA_CHECK(cudaFree(d_y));
            }

            /* ---- Mode 2: FP32  (F32 → F32, TF32 TC may activate on Ampere+) ---- */
            {
                float *d_vals, *d_x, *d_y;
                CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_x,    k   * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_y,    m   * sizeof(float)));
                CUDA_CHECK(cudaMemcpy(d_vals, vals_f32.data(),
                                      nnz * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_x, x_f32.data(),
                                      k * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(d_y, 0, m * sizeof(float)));

                cusparseSpMatDescr_t matA;
                cusparseDnVecDescr_t vecX, vecY;
                CUSPARSE_CHECK(cusparseCreateCsr(&matA, m, k, nnz,
                    d_rowPtr, d_colInd, d_vals,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
                CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, k, d_x, CUDA_R_32F));
                CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, m, d_y, CUDA_R_32F));

                const float alpha = 1.0f, beta = 0.0f;
                size_t bufSz = 0;
                CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                    handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, vecX, &beta, vecY,
                    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSz));
                void *buf = nullptr;
                if (bufSz > 0) CUDA_CHECK(cudaMalloc(&buf, bufSz));

                double ms = time_fn(stream, [&] {
                    CUSPARSE_CHECK(cusparseSpMV(
                        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, vecX, &beta, vecY,
                        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buf));
                });

                gf[F32][si][di] = 2.0 * nnz / (ms * 1e6);
                double bytes = (double)nnz   * (4 + 4)     /* vals + colInd */
                             + (double)(m+1) *  4           /* rowPtr        */
                             + (double)k     *  4           /* x fp32        */
                             + (double)m     *  4;          /* y fp32        */
                bw[F32][si][di] = bytes / (ms * 1e6);

                printf("  FP32 : %8.4f GFLOP/s  %8.3f GB/s\n",
                       gf[F32][si][di], bw[F32][si][di]);
                fflush(stdout);

                if (buf) CUDA_CHECK(cudaFree(buf));
                CUSPARSE_CHECK(cusparseDestroySpMat(matA));
                CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
                CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
                CUDA_CHECK(cudaFree(d_vals));
                CUDA_CHECK(cudaFree(d_x));
                CUDA_CHECK(cudaFree(d_y));
            }

            /* ---- Mode 3: FP64  (F64 → F64, CUDA cores) ---- */
            {
                double *d_vals, *d_x, *d_y;
                CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(double)));
                CUDA_CHECK(cudaMalloc(&d_x,    k   * sizeof(double)));
                CUDA_CHECK(cudaMalloc(&d_y,    m   * sizeof(double)));
                CUDA_CHECK(cudaMemcpy(d_vals, vals_f64.data(),
                                      nnz * sizeof(double), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_x, x_f64.data(),
                                      k * sizeof(double), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(d_y, 0, m * sizeof(double)));

                cusparseSpMatDescr_t matA;
                cusparseDnVecDescr_t vecX, vecY;
                CUSPARSE_CHECK(cusparseCreateCsr(&matA, m, k, nnz,
                    d_rowPtr, d_colInd, d_vals,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
                CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, k, d_x, CUDA_R_64F));
                CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, m, d_y, CUDA_R_64F));

                const double alpha = 1.0, beta = 0.0;
                size_t bufSz = 0;
                CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                    handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, vecX, &beta, vecY,
                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSz));
                void *buf = nullptr;
                if (bufSz > 0) CUDA_CHECK(cudaMalloc(&buf, bufSz));

                double ms = time_fn(stream, [&] {
                    CUSPARSE_CHECK(cusparseSpMV(
                        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, vecX, &beta, vecY,
                        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf));
                });

                gf[F64][si][di] = 2.0 * nnz / (ms * 1e6);
                double bytes = (double)nnz   * (8 + 4)     /* vals + colInd */
                             + (double)(m+1) *  4           /* rowPtr        */
                             + (double)k     *  8           /* x fp64        */
                             + (double)m     *  8;          /* y fp64        */
                bw[F64][si][di] = bytes / (ms * 1e6);

                printf("  FP64 : %8.4f GFLOP/s  %8.3f GB/s\n",
                       gf[F64][si][di], bw[F64][si][di]);
                fflush(stdout);

                if (buf) CUDA_CHECK(cudaFree(buf));
                CUSPARSE_CHECK(cusparseDestroySpMat(matA));
                CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
                CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
                CUDA_CHECK(cudaFree(d_vals));
                CUDA_CHECK(cudaFree(d_x));
                CUDA_CHECK(cudaFree(d_y));
            }

            CUDA_CHECK(cudaFree(d_rowPtr));
            CUDA_CHECK(cudaFree(d_colInd));
            printf("\n");
        }
    }

    /* ---- Summary tables (one GFLOP/s + one GB/s table per precision) ---- */
    for (int p = 0; p < NPREC; ++p) {
        print_table(PREC_LABEL[p], "GFLOP/s", gf[p]);
        print_table(PREC_LABEL[p], "GB/s",    bw[p]);
    }

    CUSPARSE_CHECK(cusparseDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
