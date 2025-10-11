import torch

import triton
import triton.language as tl

@triton.jit 
def _attn_fwd(
    Q, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    M,
    O,
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE = tl.constexpr,
    ):
    tl.static_assert(BLOCK_SIZE_KV <= NUM_HEADS)

    # indicates which block in the sequence length to process
    block_index_q = tl.program_id(0)

    # indicates which head and batch to process
    index_batch_head = tl.program_id(1)

    # indicates which batch this program is associated with
    index_batch = index_batch_head // NUM_HEADS

    # indicates the position of the head in the batch
    index_head = index_batch_head % NUM_HEADS

    # allows to get the (SEQ_LEN, HEAD_DIM) block in the Q, K, V by indexing it by the batch and head
    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )