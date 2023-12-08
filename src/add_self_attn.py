import torch


def add_self_attentions(attn1, lse1, attn2, lse2):
    """
    inputs:
        - attn1, attn2: 4d-tensors with shape [b, h, n, d]
        - lse1, lse2: 4d-tensors of log-sum-exp with shape [b, h, n, 1]
    output:
        - attn
        = (attn1 * exp(lse1) + attn2 * exp(lse2)) / (exp(lse1) + exp(lse2))
        = (attn1 + attn2 * exp(lse2 - lse1)) / (1 + exp(lse2-lse1))
        = attn1 * c + attn2 * (1-c), where c=1/(1 + exp(lse2-lse1)),
        - lse
        = log(exp(lse1) + exp(lse2))
        = log(exp(lse1) * (1 + exp(lse2 - lse1)))
        = lse1 + log(1 + exp(lse2 - lse1)) = lse1 - log(c)
    """
    c = (1 / (1 + (lse2 - lse1).exp())).to(dtype=attn1.dtype)
    attn = c * attn1 + (1-c) * attn2
    lse = lse1 - (c + torch.finfo(lse1.dtype).eps).log()
    return attn, lse
