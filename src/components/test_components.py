"""Unit tests for GPT component forward passes."""
import unittest

import torch

from components import feed_forward_network
from components import gpt_model
from components import multihead_attention
from components import transformer_block


class TestMultiHeadAttentionSmallBatch(unittest.TestCase):
    """Forward pass of MHA on a small hand-crafted batch."""

    def setUp(self):
        torch.manual_seed(123)
        inputs = torch.tensor(
            [[0.43, 0.15, 0.89],
             [0.55, 0.87, 0.66],
             [0.57, 0.85, 0.64],
             [0.22, 0.58, 0.33],
             [0.77, 0.25, 0.10],
             [0.05, 0.80, 0.55]])
        self.batch = torch.stack([inputs, inputs], dim=0)  # (2, 6, 3)
        cfg = multihead_attention.MultiHeadAttentionConfig(
            d_in=3, d_out=2, context_length=6, dropout=0.0, num_heads=2)
        self.mha = multihead_attention.MultiHeadAttention(cfg)

    def test_output_shape(self):
        """Test that the output shape is correct."""
        out = self.mha(self.batch)
        self.assertEqual(out.shape, torch.Size([2, 6, 2]))

    def test_output_dtype(self):
        """Test that the output dtype is correct."""
        out = self.mha(self.batch)
        self.assertEqual(out.dtype, torch.float32)

    def test_both_sequences_identical(self):
        """Both sequences in the batch are the same input, so outputs must match."""
        out = self.mha(self.batch)
        torch.testing.assert_close(out[0], out[1])


class TestGPTModelSmallForwardPass(unittest.TestCase):
    """Forward pass of a full GPT-2 small model on a two-sentence batch."""

    def setUp(self):
        torch.manual_seed(123)
        mha_cfg = multihead_attention.MultiHeadAttentionConfig(
            d_in=768, d_out=768, context_length=1024,
            dropout=0.1, num_heads=12, qkv_bias=False)
        ffn_cfg = feed_forward_network.FeedForwardNetworkConfig(emb_dim=768)
        tb_cfg = transformer_block.TransformerBlockConfig(
            mha_config=mha_cfg, ffn_config=ffn_cfg)
        cfg = gpt_model.GPTModelConfig(
            vocab_size=50257, n_layers=12, drop_out=0.1,
            transformer_block_config=tb_cfg)
        self.model = gpt_model.GPTModel(cfg)
        self.model.eval()
        self.batch = torch.tensor(
            [[6109, 3626, 6100,  345],
             [6109, 1110, 6622,  257]])

    def test_output_shape(self):
        """Test that the output shape is correct."""
        with torch.no_grad():
            out = self.model(self.batch)
        self.assertEqual(out.shape, torch.Size([2, 4, 50257]))

    def test_output_dtype(self):
        """Test that the output dtype is correct."""
        with torch.no_grad():
            out = self.model(self.batch)
        self.assertEqual(out.dtype, torch.float32)

    def test_no_nan_or_inf(self):
        """Test that the output is not NaN or Inf."""
        with torch.no_grad():
            out = self.model(self.batch)
        self.assertFalse(torch.isnan(out).any().item())
        self.assertFalse(torch.isinf(out).any().item())


if __name__ == '__main__':
    unittest.main()
