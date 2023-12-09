"""
Implementation of HyperAttention in Triton.
Tested with triton==2.1.0.

We use the FlashAttention implementation from Phil Tillet a starting point.
https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py

Requirements:
- This implementation does not support attention bias (additive mask to qk).
- This implementation only supports sequence lengths that are integer powers of two.
- the permutation indices for q and k must have the same sequence length as q and k themselves
- sequence length for q and k must be equal
"""

import math

import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
        "EVEN_V_HEADDIM": lambda args: args["v_headdim"] == args["V_BLOCK_HEADDIM"],
    }
)
# bug when seqlen_q is not divisible by BLOCK_M=128
@triton.jit
def _fwd_hyper_kernel(
        Q,
        K,
        V,
        q_sort_idx,
        k_sort_idx,
        Out,
        Lse,
        softmax_scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_q_sort_idxb,
        stride_q_sort_idxh,
        stride_q_sort_idxm,
        stride_k_sort_idxb,
        stride_k_sort_idxh,
        stride_k_sort_idxn,
        stride_ob,
        stride_oh,
        stride_om,
        nheads,
        block_size,
        sample_size,
        seqlen_k,
        seqlen_q,
        headdim,
        v_headdim,
        CACHE_KEY_SEQLEN_Q,
        CACHE_KEY_SEQLEN_K,
        BLOCK_HEADDIM: tl.constexpr,
        V_BLOCK_HEADDIM: tl.constexpr,
        EVEN_HEADDIM: tl.constexpr,
        EVEN_V_HEADDIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_vd = tl.arange(0, V_BLOCK_HEADDIM)
    # Initialize pointers to Q, K, V
    q_idx_ptrs = (
        q_sort_idx + off_b * stride_q_sort_idxb + off_h * stride_q_sort_idxh + offs_m * stride_q_sort_idxm
    )
    q_idx = tl.load(q_idx_ptrs).to(tl.int32)

    k_idx_ptrs = (k_sort_idx + off_b * stride_k_sort_idxb + off_h * stride_k_sort_idxh + offs_n * stride_k_sort_idxn)

    # initialize pointer to m and l
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, V_BLOCK_HEADDIM], dtype=tl.float32)
    q_ptrs = (
            Q + off_b * stride_qb + off_h * stride_qh + (q_idx[:, None] * stride_qm + offs_d[None, :])
    )
    if EVEN_HEADDIM:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)

    # block diagonal part
    # loop over k, v and update accumulator
    block_id = start_m // block_size
    end_n = tl.minimum((block_id + 1) * BLOCK_N * block_size, seqlen_k)
    for start_n in range(block_id * BLOCK_N * block_size, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_idx = tl.load(k_idx_ptrs + start_n * stride_k_sort_idxn).to(tl.int32)
        k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (k_idx[:, None] * stride_kn + offs_d[None, :])
        # -- compute qk ----
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
        p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator acc_o --
        acc_o = acc_o * acc_o_scale[:, None]

        v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (k_idx[:, None] * stride_vn + offs_vd[None, :])
        if EVEN_V_HEADDIM:
            v = tl.load(v_ptrs)
        else:
            v = tl.load(v_ptrs, mask=offs_vd[None, :] < v_headdim, other=0.0)
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)
    # compute sampled columns
    for col_block in range(0, sample_size):
        curr_offs_n = col_block * BLOCK_N * stride_kn + offs_n
        k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (curr_offs_n[:, None] * stride_kn + offs_d[None, :])
        # -- compute qk ----
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
        p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)
        # # -- update output accumulator acc_o --
        acc_o = acc_o * acc_o_scale[:, None]

        v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (curr_offs_n[:, None] * stride_vn + offs_vd[None, :])
        if EVEN_V_HEADDIM:
            v = tl.load(v_ptrs)
        else:
            v = tl.load(v_ptrs, mask=offs_vd[None, :] < v_headdim, other=0.0)
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)


    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]

    # initialize pointers to outputs
    lse_ptrs = Lse + off_hb * seqlen_q + q_idx
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (q_idx[:, None] * stride_om + offs_vd[None, :])
    )
    # write back l and m
    tl.store(lse_ptrs, lse_i)
    if EVEN_V_HEADDIM:
        tl.store(out_ptrs, acc_o)
    else:
        tl.store(out_ptrs, acc_o, mask=offs_vd[None, :] < v_headdim)


@triton.jit
def _bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    nheads,
    seqlen_q,
    v_headdim,
    BLOCK_M: tl.constexpr,
    V_BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, V_BLOCK_HEADDIM)
    # load
    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=offs_d[None, :] < v_headdim,
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO
        + off_b * stride_dob
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask=offs_d[None, :] < v_headdim,
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hb * seqlen_q + offs_m, delta)


@triton.jit
def _bwd_store_dx(
    dx_ptrs,
    dx,
    offs_d,
    headdim,
    even_headdim,
):
    if even_headdim:
        tl.store(dx_ptrs, dx)
    else:
        tl.store(dx_ptrs, dx, mask=offs_d[None, :] < headdim)


@triton.jit
def _bwd_blocked_kernel_one_col(
        start_n,
        Q,
        K,
        V,
        Q_idx,
        K_idx,
        DO,
        DQ,
        DK,
        DV,
        LSE,
        D,
        softmax_scale,
        stride_qm,
        stride_kn,
        stride_vn,
        stride_dom,
        stride_dqm,
        stride_dkn,
        stride_dvn,
        stride_q_idxm,
        stride_k_idxn,
        seqlen_q,
        block_size,
        headdim,
        v_headdim,
        BLOCK_HEADDIM: tl.constexpr,
        V_BLOCK_HEADDIM: tl.constexpr,
        EVEN_HEADDIM: tl.constexpr,
        EVEN_V_HEADDIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)
    block_id = start_n // block_size
    begin_m = block_id * BLOCK_M * block_size
    # initialize row / col offsets
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_vd = tl.arange(0, V_BLOCK_HEADDIM)
    # initialize pointers to value-like data
    k_idx_ptrs = K_idx + offs_n * stride_k_idxn
    k_idx = tl.load(k_idx_ptrs).to(tl.int32)
    k_ptrs = K + (k_idx[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (k_idx[:, None] * stride_vn + offs_vd[None, :])
    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, V_BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)

    # k and v stay in SRAM throughout
    if EVEN_HEADDIM:
        k = tl.load(k_ptrs)
    else:
        k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    if EVEN_V_HEADDIM:
        v = tl.load(v_ptrs)
    else:
        v = tl.load(v_ptrs, mask=offs_vd[None, :] < v_headdim, other=0.0)

    # loop over rows
    end_m = tl.minimum((block_id + 1) * BLOCK_M * block_size, seqlen_q)
    for start_m in range(begin_m, end_m, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        q_idx_ptrs = Q_idx + (start_m + offs_m) * stride_q_idxm
        q_idx = tl.load(q_idx_ptrs).to(tl.int32)
        q_ptrs = Q + (q_idx[:, None] * stride_qm + offs_d[None, :])
        # load q, k, v, do on-chip
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        # recompute p = softmax(qk, dim=-1).T
        qk = tl.dot(q, tl.trans(k))
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        lse_i = tl.load(LSE + q_idx)
        p = tl.exp(qk * softmax_scale - lse_i[:, None])
        # compute dv
        do_ptrs = DO + (q_idx[:, None] * stride_dom + offs_vd[None, :])
        if EVEN_V_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(do_ptrs, mask=offs_vd[None, :] < v_headdim, other=0.0)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        # compute dp = dot(v, do)
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        dp = tl.dot(do, tl.trans(v))
        # There's a race condition for headdim=48
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        # compute ds = p * (dp - delta[:, None])
        # Putting the subtraction after the dp matmul (instead of before) is slightly faster
        Di = tl.load(D + q_idx)
        # Converting ds to q.dtype here reduces register pressure and makes it much faster
        # for BLOCK_HEADDIM=128
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
        # compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds), q)
        # compute dq
        if not EVEN_HEADDIM:  # Otherewise there's a race condition when BIAS_TYPE='matrix'
            tl.debug_barrier()

        dq_ptrs = DQ + (q_idx[:, None] * stride_dqm + offs_d[None, :])
        dq = tl.dot(ds, k)
        if EVEN_HEADDIM:
            tl.atomic_add(dq_ptrs, dq)
        else:
            tl.atomic_add(dq_ptrs, dq, mask=offs_d[None, :] < headdim)


    # write-back
    dv_ptrs = DV + (k_idx[:, None] * stride_dvn + offs_vd[None, :])
    dk_ptrs = DK + (k_idx[:, None] * stride_dkn + offs_d[None, :])
    _bwd_store_dx(
        dk_ptrs,
        dk,
        offs_d,
        headdim,
        even_headdim=EVEN_HEADDIM,
    )
    _bwd_store_dx(
        dv_ptrs,
        dv,
        offs_vd,
        v_headdim,
        even_headdim=EVEN_V_HEADDIM,
    )



@triton.heuristics(
    {
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
        "EVEN_V_HEADDIM": lambda args: args["v_headdim"] == args["V_BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_permuted_block_diagonal_kernel(
        Q,
        K,
        V,
        q_sort_idx,
        k_sort_idx,
        DO,
        DQ,
        DK,
        DV,
        LSE,
        D,
        softmax_scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_q_sort_idxb,
        stride_q_sort_idxh,
        stride_q_sort_idxm,
        stride_k_sort_idxb,
        stride_k_sort_idxh,
        stride_k_sort_idxn,
        stride_dob,
        stride_doh,
        stride_dom,
        stride_dqb,
        stride_dqh,
        stride_dqm,
        stride_dkb,
        stride_dkh,
        stride_dkn,
        stride_dvb,
        stride_dvh,
        stride_dvn,
        nheads,
        seqlen_q,
        block_size,
        headdim,
        v_headdim,
        CACHE_KEY_SEQLEN_Q,
        CACHE_KEY_SEQLEN_K,
        BLOCK_HEADDIM: tl.constexpr,
        V_BLOCK_HEADDIM: tl.constexpr,
        EVEN_HEADDIM: tl.constexpr,
        EVEN_V_HEADDIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    Q_idx = q_sort_idx + off_b * stride_q_sort_idxb + off_h * stride_q_sort_idxh
    K_idx = k_sort_idx + off_b * stride_k_sort_idxb + off_h * stride_k_sort_idxh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    # pointer to row-wise quantities in value-like data
    D += off_hb * seqlen_q
    LSE += off_hb * seqlen_q

    start_n = tl.program_id(0)
    _bwd_blocked_kernel_one_col(
        start_n=start_n,
        Q=Q,
        K=K,
        V=V,
        Q_idx=Q_idx,
        K_idx=K_idx,
        DO=DO,
        DQ=DQ,
        DK=DK,
        DV=DV,
        LSE=LSE,
        D=D,
        softmax_scale=softmax_scale,
        stride_qm=stride_qm,
        stride_kn=stride_kn,
        stride_vn=stride_vn,
        stride_dom=stride_dom,
        stride_dqm=stride_dqm,
        stride_dkn=stride_dkn,
        stride_dvn=stride_dvn,
        stride_q_idxm=stride_q_sort_idxm,
        stride_k_idxn=stride_k_sort_idxn,
        seqlen_q=seqlen_q,
        block_size=block_size // BLOCK_N,
        headdim=headdim,
        v_headdim=v_headdim,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        V_BLOCK_HEADDIM=V_BLOCK_HEADDIM,
        EVEN_HEADDIM=EVEN_HEADDIM,
        EVEN_V_HEADDIM=EVEN_V_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )



@triton.heuristics(
    {
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
        "EVEN_V_HEADDIM": lambda args: args["v_headdim"] == args["V_BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_sampled_col_kernel(
        Q,
        K,
        V,
        DO,
        DQ,
        DK,
        DV,
        LSE,
        D,
        softmax_scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_dob,
        stride_doh,
        stride_dom,
        stride_dqb,
        stride_dqh,
        stride_dqm,
        stride_dkb,
        stride_dkh,
        stride_dkn,
        stride_dvb,
        stride_dvh,
        stride_dvn,
        nheads,
        seqlen_q,
        headdim,
        v_headdim,
        CACHE_KEY_SEQLEN_Q,
        CACHE_KEY_SEQLEN_K,
        BLOCK_HEADDIM: tl.constexpr,
        V_BLOCK_HEADDIM: tl.constexpr,
        EVEN_HEADDIM: tl.constexpr,
        EVEN_V_HEADDIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    # pointer to row-wise quantities in value-like data
    D += off_hb * seqlen_q
    LSE += off_hb * seqlen_q

    start_n = tl.program_id(0)

    # initialize row / col offsets
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_vd = tl.arange(0, V_BLOCK_HEADDIM)
    # initialize pointers to value-like data
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_vd[None, :])
    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, V_BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)

    # k and v stay in SRAM throughout
    if EVEN_HEADDIM:
        k = tl.load(k_ptrs)
    else:
        k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    if EVEN_V_HEADDIM:
        v = tl.load(v_ptrs)
    else:
        v = tl.load(v_ptrs, mask=offs_vd[None, :] < v_headdim, other=0.0)

    # loop over rows
    for start_m in range(0, seqlen_q, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        q_ptrs = Q + (offs_m_curr[:, None] * stride_qm + offs_d[None, :])
        # load q, k, v, do on-chip
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        # recompute p = softmax(qk, dim=-1).T
        qk = tl.dot(q, tl.trans(k))
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)
        p = tl.exp(qk * softmax_scale - lse_i[:, None])
        # compute dv
        do_ptrs = DO + (offs_m_curr[:, None] * stride_dom + offs_vd[None, :])
        if EVEN_V_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(do_ptrs, mask=offs_vd[None, :] < v_headdim, other=0.0)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        # compute dp = dot(v, do)
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        dp = tl.dot(do, tl.trans(v))
        # There's a race condition for headdim=48
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        # compute ds = p * (dp - delta[:, None])
        # Putting the subtraction after the dp matmul (instead of before) is slightly faster
        Di = tl.load(D + offs_m_curr)
        # Converting ds to q.dtype here reduces register pressure and makes it much faster
        # for BLOCK_HEADDIM=128
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
        # compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds), q)
        # compute dq
        if not EVEN_HEADDIM:  # Otherewise there's a race condition when BIAS_TYPE='matrix'
            tl.debug_barrier()

        dq_ptrs = DQ + (offs_m_curr[:, None] * stride_dqm + offs_d[None, :])
        dq = tl.dot(ds, k)
        if EVEN_HEADDIM:
            tl.atomic_add(dq_ptrs, dq)
        else:
            tl.atomic_add(dq_ptrs, dq, mask=offs_d[None, :] < headdim)

    dv_ptrs = DV + off_b * stride_dvb + off_h * stride_dvh + (offs_n[:, None] * stride_dvn + offs_vd[None, :])
    dk_ptrs = DK + off_b * stride_dkb + off_h * stride_dkh + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    dk += tl.load(dk_ptrs)
    dv += tl.load(dv_ptrs)
    _bwd_store_dx(
        dk_ptrs,
        dk,
        offs_d,
        headdim,
        even_headdim=EVEN_HEADDIM,
    )
    _bwd_store_dx(
        dv_ptrs,
        dv,
        offs_vd,
        v_headdim,
        even_headdim=EVEN_V_HEADDIM,
    )

    return


def _blocked_flash_attn_forward(q, k, v, q_sort_idx, k_sort_idx, block_size, sample_size, softmax_scale=None):
    """
        Initializes the forward kernel and schedules thread blocks and runs them in parallel
    """
    # shape constraints
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    _, seqlen_q_idx,_ = q_sort_idx.shape
    _, seqlen_k_idx, _ = k_sort_idx.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape[:3] == (batch, seqlen_k, nheads)
    assert q_sort_idx.shape == q.shape[:3]
    assert k_sort_idx.shape == k.shape[:3]
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda and q_sort_idx.is_cuda and k_sort_idx.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    lse = torch.empty((batch, nheads, seqlen_q), device=q.device, dtype=torch.float32)
    # o = torch.empty_like(q)
    o = torch.empty((batch, seqlen_q, nheads, v.shape[-1]), device=q.device, dtype=q.dtype)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    v_headdim = v.shape[3]
    V_BLOCK_HEADDIM = max(triton.next_power_of_2(v_headdim), 16)
    BLOCK = 128
    assert seqlen_k % BLOCK == 0, f'keys sequence length must be divisible by {BLOCK}'
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q_idx, META["BLOCK_M"]), batch * nheads)
    _fwd_hyper_kernel[grid](
        Q=q,
        K=k,
        V=v,
        q_sort_idx=q_sort_idx,
        k_sort_idx=k_sort_idx,
        Out=o,
        Lse=lse,
        softmax_scale=softmax_scale,
        stride_qb=q.stride(0),
        stride_qh=q.stride(2),
        stride_qm=q.stride(1),
        stride_kb=k.stride(0),
        stride_kh=k.stride(2),
        stride_kn=k.stride(1),
        stride_vb=v.stride(0),
        stride_vh=v.stride(2),
        stride_vn=v.stride(1),
        stride_q_sort_idxb=q_sort_idx.stride(0),
        stride_q_sort_idxh=q_sort_idx.stride(2),
        stride_q_sort_idxm=q_sort_idx.stride(1),
        stride_k_sort_idxb=k_sort_idx.stride(0),
        stride_k_sort_idxh=k_sort_idx.stride(2),
        stride_k_sort_idxn=k_sort_idx.stride(1),
        stride_ob=o.stride(0),
        stride_oh=o.stride(2),
        stride_om=o.stride(1),
        nheads=nheads,
        block_size=triton.cdiv(block_size, BLOCK),
        sample_size=triton.cdiv(sample_size, BLOCK),
        seqlen_k=seqlen_k,
        seqlen_q=seqlen_q,
        headdim=d,
        v_headdim=v_headdim,
        CACHE_KEY_SEQLEN_Q=seqlen_q // 32,
        CACHE_KEY_SEQLEN_K=seqlen_k // 32,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        V_BLOCK_HEADDIM=V_BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return o, lse, softmax_scale  # softmax_scale could have been updated


def _hyper_attn_backward(
    do, q, k, v, q_sort_idx, k_sort_idx, o, lse, dq, dk, dv, block_size, sample_size, softmax_scale=None
):
    """
    Initializes the backward kernel and schedules thread blocks and runs them in parallel
    """
    # Make sure that the last dimension is contiguous
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    # assert d in {16, 32, 64, 128}
    assert d <= 128
    assert lse.shape == (batch, nheads, seqlen_q)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == do.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    dq_accum = torch.zeros_like(q, dtype=torch.float32)
    delta = torch.empty_like(lse)

    v_headdim = v.shape[3]
    V_BLOCK_HEADDIM = max(triton.next_power_of_2(v_headdim), 16)
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](
        Out=o,
        DO=do,
        Delta=delta,
        stride_ob=o.stride(0),
        stride_oh=o.stride(2),
        stride_om=o.stride(1),
        stride_dob=do.stride(0),
        stride_doh=do.stride(2),
        stride_dom=do.stride(1),
        nheads=nheads,
        seqlen_q=seqlen_q,
        v_headdim=v_headdim,
        BLOCK_M=128,
        V_BLOCK_HEADDIM=V_BLOCK_HEADDIM,
    )

    BLOCK = 128
    num_warps = 8
    grid = lambda META: (triton.cdiv(seqlen_k, BLOCK), batch * nheads)
    _bwd_permuted_block_diagonal_kernel[grid](
        Q=q,
        K=k,
        V=v,
        q_sort_idx=q_sort_idx,
        k_sort_idx=k_sort_idx,
        DO=do,
        DQ=dq_accum,
        DK=dk,
        DV=dv,
        LSE=lse,
        D=delta,
        softmax_scale=softmax_scale,
        stride_qb=q.stride(0),
        stride_qh=q.stride(2),
        stride_qm=q.stride(1),
        stride_kb=k.stride(0),
        stride_kh=k.stride(2),
        stride_kn=k.stride(1),
        stride_vb=v.stride(0),
        stride_vh=v.stride(2),
        stride_vn=v.stride(1),
        stride_q_sort_idxb=q_sort_idx.stride(0),
        stride_q_sort_idxh=q_sort_idx.stride(2),
        stride_q_sort_idxm=q_sort_idx.stride(1),
        stride_k_sort_idxb=k_sort_idx.stride(0),
        stride_k_sort_idxh=k_sort_idx.stride(2),
        stride_k_sort_idxn=k_sort_idx.stride(1),
        stride_dob=do.stride(0),
        stride_doh=do.stride(2),
        stride_dom=do.stride(1),
        stride_dqb=dq_accum.stride(0),
        stride_dqh=dq_accum.stride(2),
        stride_dqm=dq_accum.stride(1),
        stride_dkb=dk.stride(0),
        stride_dkh=dk.stride(2),
        stride_dkn=dk.stride(1),
        stride_dvb=dv.stride(0),
        stride_dvh=dv.stride(2),
        stride_dvn=dv.stride(1),
        nheads=nheads,
        seqlen_q=seqlen_q,
        block_size=block_size,
        headdim=d,
        v_headdim=v_headdim,
        CACHE_KEY_SEQLEN_Q=seqlen_q // 32,
        CACHE_KEY_SEQLEN_K=seqlen_k // 32,  # key for triton cache (limit number of compilations)
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        V_BLOCK_HEADDIM=V_BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )

    grid = lambda META: (triton.cdiv(sample_size, BLOCK), batch * nheads)
    _bwd_sampled_col_kernel[grid](
        Q=q,
        K=k,
        V=v,
        DO=do,
        DQ=dq_accum,
        DK=dk,
        DV=dv,
        LSE=lse,
        D=delta,
        softmax_scale=softmax_scale,
        stride_qb=q.stride(0),
        stride_qh=q.stride(2),
        stride_qm=q.stride(1),
        stride_kb=k.stride(0),
        stride_kh=k.stride(2),
        stride_kn=k.stride(1),
        stride_vb=v.stride(0),
        stride_vh=v.stride(2),
        stride_vn=v.stride(1),
        stride_dob=do.stride(0),
        stride_doh=do.stride(2),
        stride_dom=do.stride(1),
        stride_dqb=dq_accum.stride(0),
        stride_dqh=dq_accum.stride(2),
        stride_dqm=dq_accum.stride(1),
        stride_dkb=dk.stride(0),
        stride_dkh=dk.stride(2),
        stride_dkn=dk.stride(1),
        stride_dvb=dv.stride(0),
        stride_dvh=dv.stride(2),
        stride_dvn=dv.stride(1),
        nheads=nheads,
        seqlen_q=seqlen_q,
        headdim=d,
        v_headdim=v_headdim,
        CACHE_KEY_SEQLEN_Q=seqlen_q // 32,
        CACHE_KEY_SEQLEN_K=seqlen_k // 32,  # key for triton cache (limit number of compilations)
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        V_BLOCK_HEADDIM=V_BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    dq.copy_(dq_accum)


class HyperAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, q_sort_idx, k_sort_idx, block_size, sample_size=0, softmax_scale=None):
        """
        q, k: queries and keys (batch_size, seqlen, nheads, headdim), seqlen must be integer power of two
        v: values (batch_size, seqlen, nheads, v_headdim)
        q_sort_idx: the permutation for queries (batch_size, seqlen, nheads)
        k_sort_idx: the permutation for keys and values (batch_size, seqlen, nheads)
        block_size: side length of block diagonal blocks
        sample_size: number of sampled columns, must be multiple of 128
        """
        # Make sure that the last dimension is contiguous
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
        assert sample_size % 128 == 0
        o, lse, ctx.softmax_scale = _blocked_flash_attn_forward(
            q, k, v, q_sort_idx, k_sort_idx, block_size, sample_size, softmax_scale=softmax_scale
        )
        ctx.save_for_backward(q, k, v, q_sort_idx, k_sort_idx, o, lse)
        ctx.block_size = block_size
        ctx.sample_size = sample_size
        return o, lse

    @staticmethod
    def backward(ctx, do, dlse_use_needed=None):
        q, k, v, q_sort_idx, k_sort_idx, o, lse = ctx.saved_tensors
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        _hyper_attn_backward(
            do,
            q,
            k,
            v,
            q_sort_idx,
            k_sort_idx,
            o,
            lse,
            dq,
            dk,
            dv,
            ctx.block_size,
            ctx.sample_size,
            softmax_scale=ctx.softmax_scale,
        )
        return dq, dk, dv, None, None, None, None, None


hyper_attn_func = HyperAttnFunc.apply

