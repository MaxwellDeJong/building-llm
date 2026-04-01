"""Unit tests for GPT component forward passes."""
import pathlib
import unittest

import dacite
import torch
import yaml

from components import feed_forward_network
from components import flash_multihead_attention
from components import gpt_model
from components import multihead_attention
from components import transformer_block

_CONFIGS_DIR = pathlib.Path(__file__).parent.parent / 'configs'


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
        ffn_cfg = feed_forward_network.FeedForwardNetworkConfig()
        tb_cfg = transformer_block.TransformerBlockConfig(
            mha_config=mha_cfg, ffn_config=ffn_cfg)
        cfg = gpt_model.GPTModelConfig(
            vocab_size=50257, n_layers=12, drop_out=0.1, emb_dim=768,
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


class TestGPTModelFromYaml(unittest.TestCase):
    """Instantiate a GPT model by deserialising a YAML config via dacite."""

    def setUp(self):
        torch.manual_seed(123)
        config_path = _CONFIGS_DIR / 'gpt2_small.yaml'
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        self.cfg = dacite.from_dict(
            data_class=gpt_model.GPTModelConfig, data=raw)
        self.model = gpt_model.GPTModel(self.cfg)
        self.model.eval()

    def test_emb_dim_propagated_to_ffn(self):
        """emb_dim specified at the top level must be propagated to ffn_config."""
        self.assertEqual(
            self.cfg.transformer_block_config.ffn_config.emb_dim,
            self.cfg.emb_dim)

    def test_output_shape(self):
        """Model built from YAML config must produce correctly shaped logits."""
        batch = torch.tensor([[6109, 3626, 6100, 345],
                               [6109, 1110, 6622, 257]])
        with torch.no_grad():
            out = self.model(batch)
        self.assertEqual(out.shape, torch.Size([2, 4, self.cfg.vocab_size]))


def _make_mha_pair(
        d_in: int,
        d_out: int,
        num_heads: int,
        context_length: int,
        block_size: int,
        seed: int = 42,
) -> tuple[multihead_attention.MultiHeadAttention,
           flash_multihead_attention.MultiHeadAttention]:
    """Return a (standard, flash) MHA pair with identical weights."""
    torch.manual_seed(seed)
    std_cfg = multihead_attention.MultiHeadAttentionConfig(
        d_in=d_in, d_out=d_out, context_length=context_length,
        dropout=0.0, num_heads=num_heads)
    std_mha = multihead_attention.MultiHeadAttention(std_cfg)
    std_mha.eval()

    flash_cfg = flash_multihead_attention.FlashAttentionConfig(
        d_in=d_in, d_out=d_out, context_length=context_length,
        dropout=0.0, num_heads=num_heads, block_size=block_size)
    flash_mha = flash_multihead_attention.MultiHeadAttention(flash_cfg)
    flash_mha.eval()

    # Mirror all learnable weights so both modules are numerically identical.
    with torch.no_grad():
        flash_mha._W_query.weight.copy_(std_mha._W_query.weight)
        flash_mha._W_key.weight.copy_(std_mha._W_key.weight)
        flash_mha._W_value.weight.copy_(std_mha._W_value.weight)
        flash_mha._out_proj.weight.copy_(std_mha._out_proj.weight)
        flash_mha._out_proj.bias.copy_(std_mha._out_proj.bias)

    return std_mha, flash_mha


class TestFlashAttentionMatchesBaseline(unittest.TestCase):
    """Flash attention output must be numerically identical to the baseline."""

    def _assert_outputs_close(
            self,
            d_in: int,
            d_out: int,
            num_heads: int,
            seq_len: int,
            context_length: int,
            block_size: int,
            batch_size: int = 2,
            seed: int = 42,
    ) -> None:
        std_mha, flash_mha = _make_mha_pair(
            d_in=d_in, d_out=d_out, num_heads=num_heads,
            context_length=context_length, block_size=block_size, seed=seed)
        torch.manual_seed(seed + 1)
        x = torch.randn(batch_size, seq_len, d_in)
        with torch.no_grad():
            out_std = std_mha(x)
            out_flash = flash_mha(x)
        torch.testing.assert_close(out_std, out_flash)

    def test_block_divides_seq_len_evenly(self):
        """block_size evenly divides sequence length."""
        self._assert_outputs_close(
            d_in=4, d_out=4, num_heads=2,
            seq_len=8, context_length=8, block_size=2)

    def test_block_does_not_divide_seq_len(self):
        """Last tile is smaller than block_size."""
        self._assert_outputs_close(
            d_in=4, d_out=4, num_heads=2,
            seq_len=7, context_length=8, block_size=3)

    def test_block_size_one(self):
        """block_size=1 gives the most fine-grained tiling."""
        self._assert_outputs_close(
            d_in=4, d_out=4, num_heads=2,
            seq_len=6, context_length=8, block_size=1)

    def test_single_tile_covers_full_sequence(self):
        """block_size >= seq_len collapses to a single tile."""
        self._assert_outputs_close(
            d_in=4, d_out=4, num_heads=2,
            seq_len=6, context_length=8, block_size=8)

    def test_single_token_sequence(self):
        """Single-token sequence: only the first query-key position matters."""
        self._assert_outputs_close(
            d_in=4, d_out=4, num_heads=2,
            seq_len=1, context_length=8, block_size=2)

    def test_larger_model_dims(self):
        """Larger d_in/d_out/num_heads combination."""
        self._assert_outputs_close(
            d_in=12, d_out=12, num_heads=4,
            seq_len=10, context_length=16, block_size=3)

    def test_output_shape_matches(self):
        """Output tensor shape must equal (batch, seq_len, d_out)."""
        std_mha, flash_mha = _make_mha_pair(
            d_in=6, d_out=6, num_heads=3,
            context_length=10, block_size=3)
        x = torch.randn(3, 9, 6)
        with torch.no_grad():
            out_std = std_mha(x)
            out_flash = flash_mha(x)
        self.assertEqual(out_std.shape, out_flash.shape)

    def test_batch_dimension_independent(self):
        """Each batch item must produce the same output regardless of other items."""
        std_mha, flash_mha = _make_mha_pair(
            d_in=4, d_out=4, num_heads=2,
            context_length=8, block_size=2)
        torch.manual_seed(0)
        x_single = torch.randn(1, 6, 4)
        x_batch = x_single.expand(3, -1, -1)
        with torch.no_grad():
            out_flash_single = flash_mha(x_single)
            out_flash_batch = flash_mha(x_batch)
        torch.testing.assert_close(
            out_flash_single.expand(3, -1, -1), out_flash_batch)


if __name__ == '__main__':
    unittest.main()
