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

    # block pointer indexing for Q, K, V, O
    # makes pointer indexing easier by defining block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset, # pointer moves from starting of Q to Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q, :]
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset, # pointer moves from starting of V to V[index_batch, index_head, :, :]
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset, # pointer moves from starting of K to K[index_batch, index_head, :, :]
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_K_dim, stride_K_seq), # invert the stride wrt Q (to transpose the matrix)
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O + qvk_offset, # pointer moves from starting of O to O[index_batch, index_head, block_index_q * BLOCK_SIZE_Q, :]
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0)
    )

    # offs_q: offsets for the tokens in the Q to process
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

    # offs_kv: offsets for the tokens in the K, V sequences to process
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    # m_i: the running max, we have one for each query
    m_i = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32) - float('inf') 

    # l_i: the running sum, we have one for each query (as we sum attention scores by rows)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0

    # acc: the accumulator for the output block, which is a group of rows of the O block
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # TODO: causal or non-causal attention O_block, l_i, m_i

    # needed for computing logsumexp (for the backward pass)
    m_i += tl.math.log(
        l_i
    )

    # normalize the block at the end, after computing all normalization factors for all rows for the current output block
    O_block = O_block / l_i[:, None]

    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q #???

    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_type)) # ??? 