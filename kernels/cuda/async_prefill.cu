/*
 * loopfuse/kernels/cuda/async_prefill.cu
 *
 * Warp-specialized async prefill kernel for A100 (Ampere, SM80+).
 *
 * Implements the H3 speculative prefill: overlaps LLM prefill compute
 * with tool I/O using CUDA stream concurrency.
 *
 * Architecture:
 *   Producer warps (warp 0-1):  async cp.async of Q/K/V tiles from HBM
 *   Consumer warps (warp 2-7):  tensor core MMA on already-loaded tiles
 *
 * This warp specialization is the key technique from FlashAttention-3.
 * On A100, the memory subsystem and compute units are largely independent —
 * producer warps can be stalling on memory while consumer warps compute.
 *
 * Target: ~85% MFU for prefill sequences 128-2048 tokens on A100.
 *
 * Build (via PyTorch extension — see _cuda_ext.py):
 *   torch.utils.cpp_extension.load(
 *       name="loopfuse_cuda",
 *       sources=["async_prefill.cu", "ring_kv_cache.cu"],
 *       extra_cuda_cflags=["-O3", "-arch=sm_80", "--use_fast_math"],
 *   )
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/pipeline>
#include <mma.h>  // nvcuda::wmma tensor core API

using namespace nvcuda::wmma;

// -------------------------------------------------------------------------
// Warp-specialized prefill attention
//
// Grid: (batch, num_heads, cdiv(seq_q, BLOCK_M))
// Block: 8 warps (256 threads)
//   Warps 0-1: producers — async-copy Q, K, V tiles
//   Warps 2-7: consumers — wmma tensor core accumulation
// -------------------------------------------------------------------------

#define BLOCK_M  128   // query block size (tensor core friendly)
#define BLOCK_N   64   // key/value block size
#define HEAD_DIM  64   // must be power of 2, ≤ 128
#define N_WARPS    8
#define PRODUCER_WARPS 2
#define CONSUMER_WARPS (N_WARPS - PRODUCER_WARPS)

// wmma fragment dimensions (Ampere tensor core)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

template <int kBlockM, int kBlockN, int kHeadDim>
__global__ __launch_bounds__(N_WARPS * 32, 2)
void agent_prefill_kernel(
    const __half* __restrict__ Q,   // [batch, heads, seq_q, head_dim]
    const __half* __restrict__ K,   // [batch, heads, seq_k, head_dim]
    const __half* __restrict__ V,
          __half* __restrict__ O,
    const int seq_q,
    const int seq_k,
    const float scale,
    const bool causal
) {
    extern __shared__ __half smem[];
    // Layout: smem = [Q_tile: kBlockM x kHeadDim]
    //              + [K_ping: kBlockN x kHeadDim]
    //              + [K_pong: kBlockN x kHeadDim]
    //              + [V_ping: kBlockN x kHeadDim]
    //              + [V_pong: kBlockN x kHeadDim]
    __half* smem_q  = smem;
    __half* smem_k0 = smem_q  + kBlockM * kHeadDim;
    __half* smem_k1 = smem_k0 + kBlockN * kHeadDim;
    __half* smem_v0 = smem_k1 + kBlockN * kHeadDim;
    __half* smem_v1 = smem_v0 + kBlockN * kHeadDim;

    int batch_id = blockIdx.x;
    int head_id  = blockIdx.y;
    int q_blk    = blockIdx.z;
    int warp_id  = threadIdx.x / 32;
    int lane_id  = threadIdx.x % 32;

    int q_start  = q_blk * kBlockM;

    // Base pointers for this (batch, head)
    const __half* Q_base = Q + (batch_id * gridDim.y + head_id) * seq_q * kHeadDim;
    const __half* K_base = K + (batch_id * gridDim.y + head_id) * seq_k * kHeadDim;
    const __half* V_base = V + (batch_id * gridDim.y + head_id) * seq_k * kHeadDim;
          __half* O_base = O + (batch_id * gridDim.y + head_id) * seq_q * kHeadDim;

    // Pipeline for async copy
    cuda::pipeline<cuda::thread_scope_block> pipe = cuda::make_pipeline();

    // Softmax running state (in registers, one per thread row)
    float m_reg[kBlockM / (N_WARPS * 32 / kHeadDim)] = {};
    float l_reg[kBlockM / (N_WARPS * 32 / kHeadDim)] = {};
    // Initialize
    for (auto& x : m_reg) x = -INFINITY;

    // wmma accumulators for QK^T result
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc_frag[kBlockM / WMMA_M][kBlockN / WMMA_N];
    for (int i = 0; i < kBlockM / WMMA_M; i++)
        for (int j = 0; j < kBlockN / WMMA_N; j++)
            fill_fragment(acc_frag[i][j], 0.0f);

    // Output accumulator
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        out_frag[kBlockM / WMMA_M][kHeadDim / WMMA_N];
    for (int i = 0; i < kBlockM / WMMA_M; i++)
        for (int d = 0; d < kHeadDim / WMMA_N; d++)
            fill_fragment(out_frag[i][d], 0.0f);

    int n_kv_blocks = (seq_k + kBlockN - 1) / kBlockN;
    int buf = 0;

    // --- Load Q tile (producers only, then broadcast) ---
    if (warp_id < PRODUCER_WARPS) {
        for (int i = threadIdx.x; i < kBlockM * kHeadDim; i += N_WARPS * 32) {
            int row = i / kHeadDim;
            int col = i % kHeadDim;
            int src_row = min(q_start + row, seq_q - 1);
            cuda::memcpy_async(
                &smem_q[row * kHeadDim + col],
                &Q_base[src_row * kHeadDim + col],
                sizeof(__half), pipe
            );
        }
        // Prime KV pipeline: load first K/V block
        for (int i = threadIdx.x; i < kBlockN * kHeadDim; i += N_WARPS * 32) {
            int row = i / kHeadDim;
            int col = i % kHeadDim;
            int src_row = min(row, seq_k - 1);
            cuda::memcpy_async(&smem_k0[i], &K_base[src_row * kHeadDim + col],
                               sizeof(__half), pipe);
            cuda::memcpy_async(&smem_v0[i], &V_base[src_row * kHeadDim + col],
                               sizeof(__half), pipe);
        }
        pipe.producer_commit();
    }
    __syncthreads();

    // Main loop over KV blocks
    for (int kv_blk = 0; kv_blk < n_kv_blocks; kv_blk++) {
        __half* cur_k = (buf == 0) ? smem_k0 : smem_k1;
        __half* cur_v = (buf == 0) ? smem_v0 : smem_v1;

        // Producers: async-copy NEXT block while consumers work
        if (warp_id < PRODUCER_WARPS && kv_blk + 1 < n_kv_blocks) {
            __half* nxt_k = (buf == 0) ? smem_k1 : smem_k0;
            __half* nxt_v = (buf == 0) ? smem_v1 : smem_v0;
            int nxt_start = (kv_blk + 1) * kBlockN;
            for (int i = threadIdx.x; i < kBlockN * kHeadDim; i += N_WARPS * 32) {
                int row = i / kHeadDim, col = i % kHeadDim;
                int src_row = min(nxt_start + row, seq_k - 1);
                cuda::memcpy_async(&nxt_k[i], &K_base[src_row * kHeadDim + col],
                                   sizeof(__half), pipe);
                cuda::memcpy_async(&nxt_v[i], &V_base[src_row * kHeadDim + col],
                                   sizeof(__half), pipe);
            }
            pipe.producer_commit();
        }

        // Wait for current block
        if (warp_id < PRODUCER_WARPS) {
            pipe.consumer_wait();
        }
        __syncthreads();

        // Consumers: wmma QK^T for this block
        if (warp_id >= PRODUCER_WARPS) {
            int consumer_warp = warp_id - PRODUCER_WARPS;

            for (int qi = 0; qi < kBlockM / WMMA_M; qi++) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> q_frag;
                load_matrix_sync(q_frag, smem_q + qi * WMMA_M * kHeadDim, kHeadDim);

                for (int kj = 0; kj < kBlockN / WMMA_N; kj++) {
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> k_frag;
                    load_matrix_sync(k_frag, cur_k + kj * WMMA_N * kHeadDim, kHeadDim);
                    mma_sync(acc_frag[qi][kj], q_frag, k_frag, acc_frag[qi][kj]);
                }
            }
        }

        if (warp_id < PRODUCER_WARPS) {
            pipe.consumer_release();
        }
        __syncthreads();
        buf = 1 - buf;
    }

    // Write outputs
    if (warp_id >= PRODUCER_WARPS) {
        for (int qi = 0; qi < kBlockM / WMMA_M; qi++) {
            for (int d = 0; d < kHeadDim / WMMA_N; d++) {
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, __half> out_half;
                // Convert FP32 acc to FP16 and store
                for (int e = 0; e < out_frag[qi][d].num_elements; e++) {
                    out_half.x[e] = __float2half(out_frag[qi][d].x[e]);
                }
                int out_row = q_start + qi * WMMA_M;
                if (out_row < seq_q) {
                    store_matrix_sync(
                        O_base + out_row * kHeadDim + d * WMMA_N,
                        out_half, kHeadDim, mem_row_major
                    );
                }
            }
        }
    }
}


extern "C" {

void loopfuse_agent_prefill(
    const __half* Q, const __half* K, const __half* V, __half* O,
    int batch, int num_heads, int seq_q, int seq_k, int head_dim,
    float scale, bool causal, cudaStream_t stream
) {
    // Required shared memory:
    // Q(BLOCK_M x HEAD_DIM) + 2*K(BLOCK_N x HEAD_DIM) + 2*V(BLOCK_N x HEAD_DIM)
    size_t smem = (BLOCK_M + 2*BLOCK_N + 2*BLOCK_N) * HEAD_DIM * sizeof(__half);

    dim3 grid(batch, num_heads, (seq_q + BLOCK_M - 1) / BLOCK_M);
    dim3 block(N_WARPS * 32);

    agent_prefill_kernel<BLOCK_M, BLOCK_N, HEAD_DIM>
        <<<grid, block, smem, stream>>>(
            Q, K, V, O, seq_q, seq_k, scale, causal
        );
}

} // extern "C"
