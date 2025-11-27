import torch

from components import multihead_attention


if __name__ == '__main__':
    torch.manual_seed(123)
    inputs = torch.tensor(
            [[0.43, 0.15, 0.89],
             [0.55, 0.87, 0.66],
             [0.57, 0.85, 0.64],
             [0.22, 0.58, 0.33],
             [0.77, 0.25, 0.10],
             [0.05, 0.80, 0.55]])
    batch = torch.stack([inputs, inputs], dim=0)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = multihead_attention.MultiHeadAttention(
            d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)

