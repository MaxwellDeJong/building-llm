import tiktoken
import torch

from components import gpt_model
from components import multihead_attention


if __name__ == '__main__':
    torch.manual_seed(123)
    #inputs = torch.tensor(
    #        [[0.43, 0.15, 0.89],
    #         [0.55, 0.87, 0.66],
    #         [0.57, 0.85, 0.64],
    #         [0.22, 0.58, 0.33],
    #         [0.77, 0.25, 0.10],
    #         [0.05, 0.80, 0.55]])
    #batch = torch.stack([inputs, inputs], dim=0)
    #batch_size, context_length, d_in = batch.shape
    #d_out = 2
    #mha = multihead_attention.MultiHeadAttention(
    #        d_in, d_out, context_length, 0.0, num_heads=2)
    #context_vecs = mha(batch)
    #print(context_vecs)

    cfg = {
        'vocab_size': 50257,
        'context_length': 1024,
        'emb_dim': 768,
        'n_heads': 12,
        'n_layers': 12,
        'drop_out': 0.1,
        'qkv_bias': False}
    model = gpt_model.GPTModel(cfg)
    batch = torch.tensor(
        [[6109, 3626, 6100, 345],
         [6109, 1110, 6622, 257]])
    print(model(batch))

