import unittest
import time
import torch
import triton
import math

import sys; sys.path.append("/home/ec2-user/workspace/hyper_attention_triton")
from src.flash_attn_triton import flash_attn_func
from src.hyper_attn_triton import hyper_attn_func
from src.attn_utils import add_self_attentions, indexing

cnt = 0


def check_memory(new_cnt=-1):
    global cnt
    if new_cnt != -1:
        cnt = new_cnt
    mem_alloc = torch.cuda.memory_allocated()/1024/1024/1024
    mem_reserve = torch.cuda.memory_reserved()/1024/1024/1024
    mem_peak = torch.cuda.memory_stats()['active_bytes.all.peak']/1024/1024/1024
    print(f"[{cnt}] mem_alloc: {mem_alloc:.4f}, mem_reserve: {mem_reserve:.4f}, mem_peak: {mem_peak:.4f}")
    cnt += 1
    return


class MyTestCase(unittest.TestCase):
    def test_block_indexing(self):
        dtype = torch.bfloat16

        batch_size = 4
        block_size = 512
        dim = 128
        head_size = 32
        seq_len = 2048
        sample_size = 128

        query = torch.randn((batch_size, head_size, seq_len, dim), device='cuda', dtype=dtype)
        key = torch.randn((batch_size, head_size, seq_len, dim), device='cuda', dtype=dtype)
        key[:, :, :sample_size, :] *= 4.
        value = torch.randn((batch_size, head_size, seq_len, dim), device='cuda', dtype=dtype)

        q_buckets_idx = torch.randint(0, 64, (batch_size, head_size, seq_len), device='cuda')
        k_buckets_idx = torch.randint(0, 64, (batch_size, head_size, seq_len), device='cuda')
        _, query_sort_idx = torch.sort(q_buckets_idx, dim=2, stable=True)
        _, key_sort_idx = torch.sort(k_buckets_idx, dim=2, stable=True)
        check_memory()

        # compute attention by presorting queries and sorting back the output attention
        t0 = time.time()
        query_sort_idx_inv = torch.argsort(query_sort_idx, dim=2, stable=True)
        query_sorted = indexing(query, query_sort_idx)
        key_sorted = indexing(key, key_sort_idx)
        value_sorted = indexing(value, key_sort_idx)
        query_split_per_block = query_sorted.view(-1, 1, block_size, dim)
        key_split_per_block = key_sorted.view(-1, 1, block_size, dim)
        value_split_per_block = value_sorted.view(-1, 1, block_size, dim)
        attn_block, lse_block = flash_attn_func(query_split_per_block.transpose(1, 2),
                                                key_split_per_block.transpose(1, 2),
                                                value_split_per_block.transpose(1, 2))

        attn_sample, lse_sample = flash_attn_func(query.transpose(1, 2),
                                                key[:, :, :sample_size, :].transpose(1, 2),
                                                value[:, :, :sample_size, :].transpose(1, 2))
        attn_block = attn_block.transpose(1, 2)
        attn_block = attn_block.view(batch_size, head_size, query_sorted.shape[2], -1)
        attn_sample = attn_sample.transpose(1, 2)
        lse_block = lse_block[:, :, :query_sorted.shape[2]]
        lse_block = lse_block.view(batch_size, head_size, query_sorted.shape[2], -1)
        flash_attn_block = indexing(attn_block, query_sort_idx_inv)+attn_sample
        lse_block = indexing(lse_block, query_sort_idx_inv)
        attn, lse = add_self_attentions(flash_attn_block, lse_block, attn_sample, lse_sample.unsqueeze(-1))
        t1 = time.time()
        check_memory()
        print('the runtime of flash attention with permutation and indexing of queries:', t1-t0)

        # torch lse computation
        qk = query_split_per_block @ key_split_per_block.transpose(-1, -2) / math.sqrt(dim)
        lse_block_torch = torch.logsumexp(qk, dim=-1, keepdim=True)
        lse_block_torch = lse_block_torch.view(batch_size, head_size, query_sorted.shape[2], -1)
        lse_block_torch = indexing(lse_block_torch, query_sort_idx_inv).squeeze(-1)
        lse_sample_torch = torch.logsumexp(
            query @ key[:, :, :sample_size, :].transpose(-1, -2) / math.sqrt(dim),
            dim=-1,
            keepdim=True
        ).squeeze(-1)
        lse_torch = (lse_sample_torch.exp() + lse_block_torch.exp()).log().to(dtype=lse_block_torch.dtype)
        print('diff between lse with sample and without: ', (lse_block_torch - lse_torch).norm(), lse_torch.norm())
        print('error flash attention:', (lse - lse_torch).norm(), lse_torch.norm())

        # compute attention kernel which permutes queries in triton
        check_memory(0)
        t2 = time.time()
        attn_triton, lse_triton = hyper_attn_func(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            query_sort_idx.transpose(1, 2),
            key_sort_idx.transpose(1, 2),
            block_size,
            sample_size,
        )
        attn_triton = attn_triton.transpose(1, 2)
        t3 = time.time()
        check_memory()

        print('the runtime of hyper attention:', t3 - t2)

        print('diff lse hyper_attention and flash with indexing and permutation: ', (lse - lse_triton).norm(), lse.norm())

        print('error hyper attention lse: ', (lse_triton - lse_torch).norm(), lse_torch.norm())

        # check if dimension of V can be different from that of Q and K
        value_small = value[:, :, :, :dim//2].clone()
        attn_triton_unequal_dim, lse_triton_unequal_dim = hyper_attn_func(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value_small.transpose(1, 2),
            query_sort_idx.transpose(1, 2),
            key_sort_idx.transpose(1, 2),
            block_size,
            sample_size,
        )
        attn_triton_unequal_dim = attn_triton_unequal_dim.transpose(1, 2)

        print('testing unequal dimension for V compared to Q, K')
        print((attn_triton[:, :, :, :dim//2] - attn_triton_unequal_dim).norm())

    def test_gradient(self):
        dtype = torch.bfloat16

        batch_size = 4
        block_size = 256
        dim = 128
        head_size = 32
        seq_len = 2048
        sample_size = 128

        query = torch.randn((batch_size, head_size, seq_len, dim), device='cuda', dtype=dtype, requires_grad=True)
        key = torch.randn((batch_size, head_size, seq_len, dim), device='cuda', dtype=dtype, requires_grad=True)
        value = torch.randn((batch_size, head_size, seq_len, dim), device='cuda', dtype=dtype, requires_grad=True)
        do = torch.randn_like(value)

        q_buckets_idx = torch.randint(0, 64, (batch_size, head_size, seq_len), device='cuda')
        k_buckets_idx = torch.randint(0, 64, (batch_size, head_size, seq_len), device='cuda')
        _, query_sort_idx = torch.sort(q_buckets_idx, dim=2, stable=True)
        _, key_sort_idx = torch.sort(k_buckets_idx, dim=2, stable=True)

        t0 = time.time()
        query_sort_idx_inv = torch.argsort(query_sort_idx, dim=2, stable=True)
        query_sorted = indexing(query, query_sort_idx)
        key_sorted = indexing(key, key_sort_idx)
        value_sorted = indexing(value, key_sort_idx)
        query_split_per_block = query_sorted.view(-1, 1, block_size, dim)
        key_split_per_block = key_sorted.view(-1, 1, block_size, dim)
        value_split_per_block = value_sorted.view(-1, 1, block_size, dim)

        attn_block, lse_block = flash_attn_func(query_split_per_block.transpose(1, 2),
                                                key_split_per_block.transpose(1, 2),
                                                value_split_per_block.transpose(1, 2))

        attn_block = attn_block.transpose(1, 2)
        attn_block = attn_block.view(batch_size, head_size, query_sorted.shape[2], -1)
        attn = indexing(attn_block, query_sort_idx_inv)

        attn.backward(do, retain_graph=True)
        t1 = time.time()
        print('flash attention and indexing forward+backward time: ', t1-t0)
        q_grad = query.grad.detach().clone()
        k_grad = key.grad.detach().clone()
        v_grad = value.grad.detach().clone()

        query.grad = None
        key.grad = None
        value.grad = None

        # torch computation

        qk = query_split_per_block @ key_split_per_block.transpose(-1, -2) / math.sqrt(dim)
        attn_block_torch = qk.softmax(dim=-1) @ value_split_per_block
        attn_torch_block = indexing(
            attn_block_torch.view(batch_size, head_size, query_sorted.shape[2], -1),
            query_sort_idx_inv
        )
        lse_block_torch = torch.logsumexp(qk, dim=-1, keepdim=True)
        lse_torch_block = indexing(
            lse_block_torch.view(batch_size, head_size, query_sorted.shape[2], -1),
            query_sort_idx_inv
        )

        qk_sample = query @ key[:, :, :sample_size, :].transpose(-1, -2) / math.sqrt(dim)
        attn_torch_sample = qk_sample.softmax(dim=-1) @ value[:, :, :sample_size, :]
        lse_torch_sample = torch.logsumexp(qk_sample, dim=-1, keepdim=True)

        attn_torch, lse_torch = add_self_attentions(attn_torch_block, lse_torch_block, attn_torch_sample, lse_torch_sample)
        lse_torch = lse_torch.squeeze(-1)
        attn_torch.backward(do, retain_graph=True)

        q_grad_torch = query.grad.detach().clone()
        k_grad_torch = key.grad.detach().clone()
        v_grad_torch = value.grad.detach().clone()

        query.grad = None
        key.grad = None
        value.grad = None

        # hyper attention computation

        t2 = time.time()
        hyper_attn, hyper_lse = hyper_attn_func(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            query_sort_idx.transpose(1, 2),
            key_sort_idx.transpose(1, 2),
            block_size,
            sample_size,
        )
        hyper_attn = hyper_attn.transpose(1, 2)
        hyper_attn.backward(do, retain_graph=True)
        t3 = time.time()
        print('hyper attention triton forward+backward time: ', t3 - t2)

        q_grad_hyper = query.grad.detach().clone()
        k_grad_hyper = key.grad.detach().clone()
        v_grad_hyper = value.grad.detach().clone()

        print('difference of torch attention and flash attn: ', (attn_torch-attn).norm(), attn_torch.norm())
        print('difference of torch lse and hyper_attention lse: ', (lse_torch - hyper_lse).norm(), lse_torch.norm())

        print('difference between gradients of queries, flash vs hyper:')
        print((q_grad - q_grad_hyper).norm())

        print('difference between gradients of keys, flash vs hyper:')
        print((k_grad - k_grad_hyper).norm())

        print('difference between gradients of values, flash vs hyper:')
        print((v_grad - v_grad_hyper).norm())

        print('difference of queries, torch vs hyper:')
        print((q_grad_torch - q_grad_hyper).norm(), q_grad_torch.norm(), q_grad_hyper.norm())

        print('difference of keys, torch vs hyper:')
        print((k_grad_torch - k_grad_hyper).norm(), k_grad_torch.norm(), k_grad_hyper.norm())

        print('difference of values, torch vs hyper:')
        print((v_grad_torch - v_grad_hyper).norm(), v_grad_torch.norm(), v_grad_hyper.norm())


if __name__ == '__main__':
    unittest.main()
    # test_runtime()