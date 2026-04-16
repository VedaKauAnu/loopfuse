/*
 * loopfuse/kernels/cuda/ring_kv_cache.cu
 *
 * Agent-optimized KV cache kernel for A100 (Ampere, SM80+).
 *
 * The central insight for agentic workloads: standard inference runtimes
 * allocate a fresh KV cache per request. An agent loop running N steps
 * means N full KV cache allocations and writes of the shared system prompt.
 * This kernel implements a RING BUFFER KV cache that:
 *
 *   1. Persists across agent steps (no re-allocation)
 *   2. Uses CUDA async memcpy (cp.async) to prefetch the next slot while
 *      the current step's compute is running
 *   3. Exposes an async prefetch API so the SpecPrefillPass can overlap
 *      KV cache prep with tool I/O
 *
 * Roofline target: memory-bound (KV cache access is ~0.5 FLOP/byte).
 * Primary optimization axis: HBM bandwidth utilization, not compute.
 * Target: >85% of peak HBM bandwidth on A100 (2 TB/s).
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -std=c++17 ring_kv_cache.cu -o ring_kv_cache.so
 *   Or via torch.utils.cpp_extension.load() in Python.
 *
 * Python binding: see loopfuse/kernels/cuda/_ring_kv_cache_binding.py
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <assert.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------

#define MAX_LAYERS   64
#define MAX_HEADS    64
#define MAX_HEAD_DIM 256
#define WARP_SIZE    32

// -------------------------------------------------------------------------
// KV Cache descriptor (passed to all kernels)
// -------------------------------------------------------------------------

struct KVCacheDesc {
    __half* keys;         // [capacity, num_layers, num_heads, head_dim]
    __half* values;       // [capacity, num_layers, num_heads, head_dim]
    int     write_head;   // ring buffer write pointer
    int     capacity;     // total slots (agent steps)
    int     num_layers;
    int     num_heads;
    int     head_dim;
    int     seq_len;      // current sequence length
};

// -------------------------------------------------------------------------
// Kernel 1: Write new KV entries to the ring buffer slot.
//
// Called after each LLM forward pass to commit the new keys/values.
// Uses vectorized FP16 stores (128-bit transactions) to saturate HBM write.
// -------------------------------------------------------------------------

__global__ void ring_kv_write_kernel(
    __half* __restrict__ cache_keys,    // ring buffer keys
    __half* __restrict__ cache_values,  // ring buffer values
    const __half* __restrict__ new_keys,    // [num_layers, num_heads, seq_new, head_dim]
    const __half* __restrict__ new_values,
    int slot_idx,       // ring buffer slot to write
    int num_layers,
    int num_heads,
    int seq_new,        // number of new tokens to write
    int head_dim,
    int max_seq_len     // ring buffer capacity per slot
) {
    // Each thread writes 8 FP16 values (128-bit) at once
    const int VEC = 8;
    using Vec8 = __align__(16) __half[VEC];

    int layer   = blockIdx.z;
    int head    = blockIdx.y;
    int tok_vec = blockIdx.x * blockDim.x + threadIdx.x;  // vectorized token position

    int total_vecs = (seq_new * head_dim) / VEC;
    if (tok_vec >= total_vecs) return;

    // Source: new_keys[layer, head, :, :]
    int src_offset = (layer * num_heads * seq_new * head_dim
                    + head  * seq_new * head_dim
                    + tok_vec * VEC);

    // Destination: cache_keys[slot, layer, head, seq_pos, :]
    // We write to the slot_idx'th position in the ring
    int dst_offset = (slot_idx   * num_layers * num_heads * max_seq_len * head_dim
                    + layer      * num_heads  * max_seq_len * head_dim
                    + head       * max_seq_len * head_dim
                    + tok_vec    * VEC);

    // 128-bit store: 8 FP16 values at once
    *reinterpret_cast<Vec8*>(cache_keys + dst_offset) =
        *reinterpret_cast<const Vec8*>(new_keys + src_offset);
    *reinterpret_cast<Vec8*>(cache_values + dst_offset) =
        *reinterpret_cast<const Vec8*>(new_values + src_offset);
}

// -------------------------------------------------------------------------
// Kernel 2: Decode attention reading from the ring buffer KV cache.
//
// Memory-optimized for decode (batch=1, seq_q=1):
//   - Uses cp.async for pipelined K/V loading
//   - Double-buffered shared memory (one buffer loading, one computing)
//   - FP16 accumulator (safe for decode, avoids expensive FP32 upcast)
// -------------------------------------------------------------------------

template <int HEAD_DIM, int BLOCK_N>
__global__ void ring_kv_decode_kernel(
    const __half* __restrict__ Q,       // [num_heads, 1, head_dim]
    const __half* __restrict__ K_ring,  // ring buffer keys
    const __half* __restrict__ V_ring,
    __half* __restrict__ O,             // [num_heads, 1, head_dim]
    int head_idx,
    int seq_len,     // total sequence length in cache
    int num_layers,
    int num_heads,
    float scale
) {
    // Double-buffered shared memory for K and V
    __shared__ __half smem_k[2][BLOCK_N][HEAD_DIM];
    __shared__ __half smem_v[2][BLOCK_N][HEAD_DIM];

    // Softmax accumulators in registers
    float acc[HEAD_DIM / WARP_SIZE] = {0.0f};
    float m_i = -INFINITY;
    float l_i = 0.0f;

    // Load Q into registers
    float q_reg[HEAD_DIM / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM / WARP_SIZE; d++) {
        q_reg[d] = __half2float(Q[head_idx * HEAD_DIM + lane + d * WARP_SIZE]) * scale;
    }

    int n_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    int buf = 0;  // double buffer index

    // Pipeline: prefetch first block
    auto pipe = cuda::make_pipeline();

    // Prime the pipeline: async-copy block 0
    if (0 < n_blocks) {
        int k_base = head_idx * seq_len * HEAD_DIM;
        for (int i = threadIdx.x; i < BLOCK_N * HEAD_DIM; i += blockDim.x) {
            int pos = i / HEAD_DIM;
            int dim = i % HEAD_DIM;
            int src_pos = min(pos, seq_len - 1);
            cuda::memcpy_async(
                &smem_k[0][pos][dim],
                &K_ring[k_base + src_pos * HEAD_DIM + dim],
                sizeof(__half), pipe
            );
            cuda::memcpy_async(
                &smem_v[0][pos][dim],
                &V_ring[k_base + src_pos * HEAD_DIM + dim],
                sizeof(__half), pipe
            );
        }
        pipe.producer_commit();
    }

    for (int blk = 0; blk < n_blocks; blk++) {
        // Prefetch NEXT block while computing CURRENT
        int next_buf = 1 - buf;
        if (blk + 1 < n_blocks) {
            int k_base = head_idx * seq_len * HEAD_DIM + (blk + 1) * BLOCK_N * HEAD_DIM;
            for (int i = threadIdx.x; i < BLOCK_N * HEAD_DIM; i += blockDim.x) {
                int pos = i / HEAD_DIM;
                int dim = i % HEAD_DIM;
                int src_pos = min((blk + 1) * BLOCK_N + pos, seq_len - 1);
                cuda::memcpy_async(
                    &smem_k[next_buf][pos][dim],
                    &K_ring[head_idx * seq_len * HEAD_DIM + src_pos * HEAD_DIM + dim],
                    sizeof(__half), pipe
                );
                cuda::memcpy_async(
                    &smem_v[next_buf][pos][dim],
                    &V_ring[head_idx * seq_len * HEAD_DIM + src_pos * HEAD_DIM + dim],
                    sizeof(__half), pipe
                );
            }
            pipe.producer_commit();
        }

        // Wait for current block to be ready
        pipe.consumer_wait();
        __syncthreads();

        // Compute QK^T scores for this block
        int blk_start = blk * BLOCK_N;
        #pragma unroll 4
        for (int n = 0; n < BLOCK_N; n++) {
            int pos = blk_start + n;
            if (pos >= seq_len) break;

            // Dot product Q·K[n] across warps
            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM / WARP_SIZE; d++) {
                score += q_reg[d] * __half2float(smem_k[buf][n][lane + d * WARP_SIZE]);
            }
            // Warp reduction
            #pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
                score += __shfl_xor_sync(0xffffffff, score, mask);

            // Online softmax update (only thread 0 of warp is correct after reduction)
            if (lane == 0) {
                float m_new = fmaxf(m_i, score);
                float alpha = expf(m_i - m_new);
                float p     = expf(score - m_new);
                #pragma unroll
                for (int d = 0; d < HEAD_DIM / WARP_SIZE; d++) {
                    acc[d] = alpha * acc[d] +
                             p * __half2float(smem_v[buf][n][lane + d * WARP_SIZE]);
                }
                l_i = alpha * l_i + p;
                m_i = m_new;
            }
        }

        pipe.consumer_release();
        __syncthreads();
        buf = next_buf;
    }

    // Normalize and write output
    if (lane == 0) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM / WARP_SIZE; d++) {
            O[head_idx * HEAD_DIM + d * WARP_SIZE] = __float2half(acc[d] / l_i);
        }
    }
}

// -------------------------------------------------------------------------
// Kernel 3: Async KV cache slot zeroing.
//
// Called by SpecPrefillPass: while the tool call is running (GPU would be
// idle), we pre-zero the next ring buffer slot so it's ready for the
// speculative prefill. Overlaps with CPU-side tool I/O.
// -------------------------------------------------------------------------

__global__ void async_zero_next_slot_kernel(
    __half* __restrict__ cache_keys,
    __half* __restrict__ cache_values,
    int next_slot,
    int slot_size_halves  // num_layers * num_heads * max_seq_len * head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = next_slot * slot_size_halves;

    // Vectorized zero: 8 FP16 at once
    const int VEC = 8;
    int n_vecs = slot_size_halves / VEC;
    if (idx < n_vecs) {
        *reinterpret_cast<float4*>(cache_keys  + base + idx * VEC) = {0,0,0,0};
        *reinterpret_cast<float4*>(cache_values + base + idx * VEC) = {0,0,0,0};
    }
}

// -------------------------------------------------------------------------
// Host-callable C interface (used by Python ctypes binding)
// -------------------------------------------------------------------------

extern "C" {

void loopfuse_ring_kv_write(
    __half* cache_keys, __half* cache_values,
    const __half* new_keys, const __half* new_values,
    int slot_idx, int num_layers, int num_heads,
    int seq_new, int head_dim, int max_seq_len,
    cudaStream_t stream
) {
    int total_elems = num_layers * num_heads * seq_new * head_dim;
    int threads = 256;
    int vecs    = (total_elems / 8 + threads - 1) / threads;
    dim3 grid(vecs, num_heads, num_layers);
    ring_kv_write_kernel<<<grid, threads, 0, stream>>>(
        cache_keys, cache_values, new_keys, new_values,
        slot_idx, num_layers, num_heads, seq_new, head_dim, max_seq_len
    );
}

void loopfuse_async_zero_slot(
    __half* cache_keys, __half* cache_values,
    int next_slot, int num_layers, int num_heads,
    int max_seq_len, int head_dim,
    cudaStream_t stream
) {
    int slot_size = num_layers * num_heads * max_seq_len * head_dim;
    int threads   = 256;
    int blocks    = (slot_size / 8 + threads - 1) / threads;
    async_zero_next_slot_kernel<<<blocks, threads, 0, stream>>>(
        cache_keys, cache_values, next_slot, slot_size
    );
}

} // extern "C"
