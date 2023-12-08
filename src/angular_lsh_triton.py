import math
import torch
import triton
import triton.language as tl


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen"] % args["BLOCK_M"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _angular_lsh_kernel(
        in_mat,
        proj_dir,
        perm,
        enc_vec,
        buckets,
        stride_in_matb,
        stride_in_math,
        stride_in_matm,
        stride_proj_dirb,
        stride_proj_dirh,
        stride_proj_dird,
        stride_bucketsb,
        stride_bucketsh,
        nheads,
        seqlen,
        seqlen_rounded,
        headdim,
        NUM_PROJ_ROUNDED: tl.constexpr,
        num_projs: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_HEADDIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, NUM_PROJ_ROUNDED)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    in_mat_ptrs = (
            in_mat + off_b * stride_in_matb + off_h * stride_in_math + (offs_m[:, None] * stride_in_matm +
                                                                        offs_d[None, :])
    )
    proj_dir_ptrs = (
        proj_dir + off_b * stride_proj_dirb + off_h * stride_proj_dirh + (offs_d[:, None] * stride_proj_dird +
                                                                          offs_n[None, :])
    )

    # load in_mat block
    if EVEN_M:
        if EVEN_HEADDIM:
            mat = tl.load(in_mat_ptrs)
        else:
            mat = tl.load(in_mat_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            mat = tl.load(in_mat_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
        else:
            mat = tl.load(in_mat_ptrs, mask=(offs_m[:, None] < seqlen) & (offs_d[None, :] < headdim), other=0.0)

    # load proj_dir block, need to mask out out of bound offsets
    if EVEN_HEADDIM:
        proj_dir_block = tl.load(proj_dir_ptrs, mask=offs_n[None, :] < num_projs, other=0.0)
    else:
        proj_dir_block = tl.load(proj_dir_ptrs,
                                 mask=(offs_n[None, :] < num_projs) & (offs_d[:, None] * stride_proj_dird < headdim),
                                 other=0.0)

    # multiply the in_mat block with proj_dir block to get the mask
    mask = tl.dot(mat, proj_dir_block)
    mask = tl.where(mask > 0.0, 1.0, 0.0)

    # form enc_vec
    encoding_vectors = tl.load(enc_vec+offs_n, mask=offs_n < num_projs, other=0.0)

    # multiply mask by enc_vec
    bin_ids = tl.sum(mask * encoding_vectors[None, :], 1).to(tl.int32)
    # bin_ids = tl.ravel(bin_ids)  # flatten the bin_ids into a 1d tensor

    # read hash buckets from look up table
    hash_buckets = tl.load(perm+bin_ids)

    # write back bin_ids
    # initialize pointers to output
    buckets_ptrs = buckets + off_b * stride_bucketsb + off_h * stride_bucketsh + offs_m
    if EVEN_M:
        tl.store(buckets_ptrs, hash_buckets)
    else:
        tl.store(buckets_ptrs, hash_buckets, mask=offs_m < seqlen)


def _angular_lsh(in_mat, proj_dir, perm, enc_vec):
    # shape constraints
    num_projs = proj_dir.shape[-1]
    batch, nheads, seqlen, d = in_mat.shape
    assert (proj_dir.shape == (batch, nheads, d, num_projs)) or (proj_dir.shape == (1, 1, d, num_projs))
    assert in_mat.dtype == proj_dir.dtype, "All three tensors must have the same type"
    assert in_mat.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert in_mat.is_cuda and proj_dir.is_cuda and perm.is_cuda and enc_vec.is_cuda
    if proj_dir.shape[:2] == (1, 1):
        stride_proj_dirb, stride_proj_dirh = 0, 0
    else:
        stride_proj_dirb, stride_proj_dirh = proj_dir.stride()[:2]

    seqlen_rounded = math.ceil(seqlen / 128) * 128
    num_projs_rounded = 16
    buckets = torch.empty((batch, nheads, seqlen), device=in_mat.device, dtype=torch.int32)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch * nheads)
    _angular_lsh_kernel[grid](
        in_mat=in_mat,
        proj_dir=proj_dir,
        perm=perm,
        enc_vec=enc_vec,
        buckets=buckets,
        stride_in_matb=in_mat.stride(0),
        stride_in_math=in_mat.stride(1),
        stride_in_matm=in_mat.stride(2),
        stride_proj_dirb=stride_proj_dirb,
        stride_proj_dirh=stride_proj_dirh,
        stride_proj_dird=proj_dir.stride(2),
        stride_bucketsb=buckets.stride(0),
        stride_bucketsh=buckets.stride(1),
        nheads=nheads,
        seqlen=seqlen,
        seqlen_rounded=seqlen_rounded,
        headdim=d,
        NUM_PROJ_ROUNDED=num_projs_rounded,
        num_projs=num_projs,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return buckets


class AngularLSHTriton(torch.nn.Module):
    """
    inputs:
        - num_projs: a positive integer that determines the number of random projections used by hash function
        - dim: positive integer that determines the dimension of input vectors
        - mat: a tensor whose last shape is equal to dim and gets hashed by the lsh function
    output:
        - buckets: a tensor with shape mat.shape[:-1] and each entry is an integer in [0, 2^num_proj - 1]
    """
    def __init__(self, num_projs, dim, rng=None):
        super().__init__()
        self.num_projs = num_projs

        if num_projs > 0:
            self.register_buffer('perm', self._unit_hamming_distance_array(self.num_projs), persistent=False)
            self.register_buffer('proj_dir', torch.randn(dim + (num_projs,), generator=rng), persistent=False)
            self.register_buffer('enc_vec', 2 ** torch.arange(self.num_projs).view(1, 1, 1, -1), persistent=False)
        else:
            raise ValueError("Invalid value for num_projs")

    def _unit_hamming_distance_array(self, size_n):
        if size_n == 1:
            return torch.tensor([0, 1], dtype=torch.int32)
        a = self._unit_hamming_distance_array(size_n - 1)
        b = torch.concat([a, torch.flip(a, dims=[0]) + 2 ** (size_n - 1)], 0)
        return b if b.stride(-1) == 1 else b.contiguous()

    def hash_torch(self, mat):
        mask = torch.einsum('...nd,...dr -> ...nr', mat, self.proj_dir)
        mask = mask > 0
        bin_ids = (mask * self.enc_vec).sum(-1)
        return self.perm[bin_ids]

    def hash_triton(self, mat):
        return _angular_lsh(mat, self.proj_dir, self.perm, self.enc_vec)

    def __repr__(self):
        return f"AngularLSH(num_proj={self.num_projs}, proj_dir.shape={self.proj_dir.shape})"
