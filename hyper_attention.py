import torch

from src.attn_utils import add_self_attentions
from src.flash_attn_triton import flash_attn_func
from src.hyper_attn_triton import hyper_attn_func
from src.angular_lsh_triton import AngularLSHTriton


class HyperAttention(torch.nn.Module):

    def __init__(self, input_dim=64, lsh_num_projs=8, block_size=256, sample_size=256, min_seq_len=2048, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.lsh_num_projs = lsh_num_projs
        self.block_size = block_size
        self.sample_size = sample_size
        self.min_seq_len = min_seq_len
        self.lsh = AngularLSHTriton(num_projs=self.lsh_num_projs, dim=(1, 1, input_dim))

    def forward(self, query: torch.tensor, key: torch.tensor, value: torch.tensor, scale=None, causal=False,
                return_lse=False):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        n_query = query.shape[2]
        batch_size, n_heads, n_key, dim = key.shape
        scale = scale or dim ** (-0.5)
        assert n_query == n_key

        # without causal masking
        if causal is False:
            attn, lse = self.forward_no_causal_mask(query, key, value, scale)

        else:  # with causal masking
            if n_key <= self.min_seq_len:
                attn, lse = flash_attn_func(query.transpose(1, 2),
                                            key.transpose(1, 2),
                                            value.transpose(1, 2),
                                            None, True, scale)
                attn = attn.transpose(1, 2)

            else:
                # If n_query is odd we pad inputs by zero rows
                if n_query % 2:
                    query = torch.nn.functional.pad(query, (0, 0, 0, 1), mode='constant', value=0.)
                    key = torch.nn.functional.pad(key, (0, 0, 0, 1), mode='constant', value=0.)
                    value = torch.nn.functional.pad(value, (0, 0, 0, 1), mode='constant', value=0.)

                # extract block diagonal parts
                q_bd = query.view(batch_size, 2 * n_heads, query.shape[2] // 2, query.shape[-1])
                k_bd = key.view(batch_size, 2 * n_heads, key.shape[2] // 2, key.shape[-1])
                v_bd = value.view(batch_size, 2 * n_heads, key.shape[2] // 2, value.shape[-1])

                attn_bd, lse_bd = self.forward(q_bd, k_bd, v_bd, scale, True, True)

                if attn_bd.shape[2] not in attn_bd.stride():
                    attn_bd = attn_bd.contiguous()
                attn_bd = attn_bd.view(batch_size, n_heads, -1, dim)

                if lse_bd.shape[2] not in lse_bd.stride():
                    lse_bd = lse_bd.contiguous()
                lse_bd = lse_bd.view(batch_size, n_heads, -1, 1)

                # lowe diagonal block is an unmasked attention
                attn_unmasked, lse_unmasked = self.forward_no_causal_mask(
                    query[:, :, key.shape[2] // 2:, :], key[:, :, :key.shape[2] // 2, :],
                    value[:, :, :key.shape[2] // 2, :], scale)

                attn_up, lse_up = attn_bd[:, :, :query.shape[2] // 2, :], lse_bd[:, :, :query.shape[2] // 2, :]
                attn_down, lse_down = add_self_attentions(attn_bd[:, :, query.shape[2] // 2:, :],
                                                          lse_bd[:, :, query.shape[2] // 2:, :],
                                                          attn_unmasked, lse_unmasked)

                attn = torch.cat((attn_up, attn_down), dim=-2)
                lse = torch.cat((lse_up, lse_down), dim=-2)

                if n_query % 2:
                    attn = attn[:, :, :-1, :]
                    lse = lse[:, :, :-1, :]

        if not return_lse:
            return attn
        else:
            return attn, lse

    def forward_no_causal_mask(self, query, key, value, scale):

        batch_size, head_size, n_query, dim = query.shape

        if self.min_seq_len > n_query:
            return flash_attn_func(query, key, value, None, False, scale)

        # Hash keys and queries via SortLSH and obtain buckets
        _, query_sort_idx = torch.sort(self.lsh.hash_triton(query), dim=2, stable=True)  # batch_size x head_size x n
        _, key_sort_idx = torch.sort(self.lsh.hash_triton(key), dim=2, stable=True)

        # Now run hyper attention function on q,k,v and the permutations
        attn, lse = hyper_attn_func(query.transpose(1, 2),
                                    key.transpose(1, 2),
                                    value.transpose(1, 2),
                                    query_sort_idx.transpose(1, 2),
                                    key_sort_idx.transpose(1, 2),
                                    self.block_size,
                                    self.sample_size,
                                    scale,
                                    )

        attn = attn.transpose(1, 2)

        return attn, lse.unsqueeze(-1)
