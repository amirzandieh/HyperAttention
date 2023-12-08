# HyperAttention: Long-context Attention in Near-Linear Time

Triton Implementation of HyperAttention Algorithm

# Requirements

The code requires ``pytorch`` and [``triton``](https://github.com/openai/triton).
pytorch version 2.0.1 tested, but any version >= 2.0.0 might work.
Also makes use of [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main) in [triton](https://github.com/openai/triton) implementation. Flash attention kernel adapted to work with triton version **2.1.0.**

# How to use

The impelmentation of HyperAttention can be found in ``hyper_attention.py``. An example of usage:

```python
from hyper_attention import HyperAttention

attn = HyperAttention(
    input_dim=64, 
    lsh_num_projs=7,
    block_size=256,
    sample_size=256,
    min_seq_len=2048)

attn_output = attn(query, key, value, causal=True)
```

The module has the following parameters:
- ```input_dim```: the dimension of input query and key. (Required)
- ```lsh_num_projs```: the number of random projection vectors used in the locality sensitive hashing scheme. The default is 7.
- ```block_size```: the size of blocks for the block-diagonal approximation. The default is 256.
- ```sample_size```: the number of sampled columns in the attention matrix $A$. The default is 256.
- ```min_seq_len```: minimum sequence length that HyperAttention applies. When the sequence length is smaller than this value we compute exactly using the FlashAttention because of overheads of HyperAttention may dominate the runtim. The default value is ```2048```.
