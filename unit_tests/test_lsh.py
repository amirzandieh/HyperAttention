import unittest
import time
import torch
import triton

import sys;

sys.path.append("/home/ec2-user/workspace/hyper_attention_triton")
from src.angular_lsh_triton import AngularLSHTriton

cnt = 0

def check_memory():
    global cnt
    mem_alloc = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    mem_reserve = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
    mem_peak = torch.cuda.memory_stats()['active_bytes.all.peak'] / 1024 / 1024 / 1024
    print(f"[{cnt}] mem_alloc: {mem_alloc:.4f}, mem_reserve: {mem_reserve:.4f}, mem_peak: {mem_peak:.4f}")
    cnt += 1


class MyTestCase(unittest.TestCase):
    def test_1_validation(self):
        print("1. this is validation test")
        dtype = torch.float16
        block_size, dim, batch_size, head_size, seq_len = 256, 128, 4, 32, 2048
        num_projs = 8

        query = torch.randn((batch_size, head_size, seq_len, dim), device='cuda', dtype=dtype)

        self.lsh = AngularLSHTriton(num_projs=num_projs, dim=(1, 1, dim)).to(device='cuda', dtype=dtype)

        # apply lsh in pytorch
        t0 = time.time()
        query_hash_buckets = self.lsh.hash_torch(query)
        t1 = time.time()
        print('the runtime of torch lsh:', t1 - t0)

        # apply lsh in triton
        check_memory()
        t2 = time.time()
        query_hash_buckets_triton = self.lsh.hash_triton(query)
        t3 = time.time()
        check_memory()
        print('the runtime of triton lsh:', t3 - t2)

        print('difference between torch and triton hashes: ', (query_hash_buckets.float() - query_hash_buckets_triton.float()).norm())

    def test_2_runtime(self):
        print()
        print("2. this is runtime test")

        block_size, dim, batch_size, head_size = 256, 128, 4, 32
        num_projs = 8
        seq_len = 2048
        dtype = torch.float16

        query = torch.randn((batch_size, head_size, seq_len, dim), device='cuda', dtype=dtype)

        self.lsh = AngularLSHTriton(num_projs=num_projs, dim=(1, 1, dim)).to(device='cuda', dtype=dtype)

        def test_fn1():
            query_hash_buckets = self.lsh.hash_torch(query)

        warmup = 20
        rep = 1000

        tim_py_q20, tim_py_q50, tim_py_q80 = triton.testing.do_bench(test_fn1, warmup=warmup, rep=rep,
                                                                     quantiles=[0.2, 0.5, 0.8])
        print(f"pytorch runtime: {tim_py_q50:.5f} ms ({tim_py_q20:.5f}, {tim_py_q80:.5f})")

        def test_fn2():
            query_hash_buckets_triton = self.lsh.hash_triton(query)

        tim_tr_q20, tim_tr_q50, tim_tr_q80 = triton.testing.do_bench(test_fn2, warmup=warmup, rep=rep,
                                                                     quantiles=[0.2, 0.5, 0.8])
        print(f"triton  runtime: {tim_tr_q50:.5f} ms ({tim_tr_q20:.5f}, {tim_tr_q80:.5f})")


if __name__ == '__main__':
    unittest.main()
