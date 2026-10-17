import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _CountedQueryStartLoc:
    def __init__(self, length):
        self.length = length
        self.reads = 0

    def __getitem__(self, index):
        assert index == -1
        self.reads += 1
        return self.length


class TestKDALogicalNumTokens(unittest.TestCase):
    def _metadata(self, lengths, *, cpu_mirror=True, mode=ForwardMode.EXTEND):
        backend = MambaAttnBackendBase.__new__(MambaAttnBackendBase)
        backend.device = "cpu"
        backend.topk = 0
        backend.req_to_token_pool = SimpleNamespace(
            get_mamba_indices=lambda indices: indices,
            translate_mamba_indices=lambda indices: indices,
            mamba_pool=None,
        )
        starts = [sum(lengths[:i]) for i in range(len(lengths))]
        batch = SimpleNamespace(
            batch_size=len(lengths),
            req_pool_indices=torch.arange(len(lengths)),
            mamba_track_indices=None,
            mamba_track_mask=None,
            _original_batch_size=None,
            forward_mode=mode,
            extend_start_loc=torch.tensor(starts, dtype=torch.int32),
            extend_seq_lens=torch.tensor(lengths, dtype=torch.int32),
            extend_seq_lens_cpu=lengths if cpu_mirror else None,
            # Deliberately padded: this is not the logical length.
            extend_num_tokens=sum(lengths) + 16,
            input_ids=torch.zeros(sum(lengths), dtype=torch.int64),
            spec_info=SimpleNamespace(ragged_verify_layout=None, draft_token_num=3),
        )
        return backend._forward_metadata(batch)

    def test_host_lengths_cover_ragged_and_padded_batches(self):
        for lengths in ([2048], [1024], [5, 0, 3], [0]):
            with self.subTest(lengths=lengths):
                metadata = self._metadata(lengths)
                expected = int(metadata.query_start_loc[-1])
                query = _CountedQueryStartLoc(expected)
                metadata.query_start_loc = query
                self.assertEqual(metadata.logical_num_tokens, expected)
                for _ in range(34):
                    self.assertEqual(metadata.get_logical_num_tokens(), expected)
                self.assertEqual(query.reads, 0)

    def test_gpu_only_fallback_reads_once_per_forward(self):
        for length in (2048, 1024, 0):
            with self.subTest(length=length):
                metadata = self._metadata([length], cpu_mirror=False)
                query = _CountedQueryStartLoc(length)
                metadata.query_start_loc = query
                for _ in range(34):
                    self.assertEqual(metadata.get_logical_num_tokens(), length)
                self.assertEqual(query.reads, 1)

    def test_new_forward_does_not_reuse_previous_length(self):
        for lengths in ([2048], [1024], [3, 7], [2048]):
            self.assertEqual(
                self._metadata(lengths).get_logical_num_tokens(), sum(lengths)
            )

    def test_decode_and_verify_do_not_cache_extend_lengths(self):
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            with self.subTest(mode=mode):
                metadata = self._metadata([3, 3], mode=mode)
                self.assertIsNone(metadata.logical_num_tokens)

    def test_padding_is_trimmed_before_kernels_and_restored_as_zeros(self):
        for logical, physical in ((3, 8), (3, 3)):
            with self.subTest(logical=logical, physical=physical):
                query = _CountedQueryStartLoc(logical)
                backend = KDAAttnBackend.__new__(KDAAttnBackend)
                backend.forward_metadata = ForwardMetadata(
                    query_start_loc=query,
                    mamba_cache_indices=torch.tensor([0]),
                    logical_num_tokens=logical,
                )
                backend.req_to_token_pool = SimpleNamespace(
                    mamba2_layer_cache=lambda _: SimpleNamespace(
                        conv=[torch.zeros(1, 3, 6)],
                        temporal=torch.zeros(1, 1, 2, 2),
                    )
                )
                backend.accept_lens_pool = None
                backend.kernel_dispatcher = SimpleNamespace(
                    extend=Mock(side_effect=lambda **kw: kw["v"].clone())
                )
                layer = SimpleNamespace(
                    layer_id=0,
                    q_dim=2,
                    k_dim=2,
                    v_dim=2,
                    head_q_dim=2,
                    head_k_dim=2,
                    head_v_dim=2,
                    conv_weights=None,
                    bias=None,
                    A_log=None,
                    dt_bias=None,
                    lower_bound=None,
                )
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    extend_prefix_lens=torch.tensor([0]),
                    extend_seq_lens_cpu=[logical],
                )
                x = torch.arange(physical * 6, dtype=torch.float32).reshape(physical, 6)
                with patch(
                    "sglang.srt.layers.attention.linear.kda_backend.causal_conv1d_fn",
                    side_effect=lambda value, *args, **kw: value,
                ) as conv:
                    output = backend.forward_extend(
                        layer,
                        batch,
                        x,
                        torch.ones(1, physical, 2),
                        torch.ones(1, physical, 1),
                    )
                self.assertEqual(conv.call_args.args[0].shape, (6, logical))
                args = backend.kernel_dispatcher.extend.call_args.kwargs
                self.assertEqual(args["g"].shape[1], logical)
                self.assertEqual(args["beta"].shape[1], logical)
                self.assertEqual(output.shape, (1, physical, 1, 2))
                self.assertTrue(torch.equal(output[0, :logical, 0], x[:logical, 4:]))
                self.assertEqual(torch.count_nonzero(output[:, logical:]).item(), 0)
                self.assertEqual(query.reads, 0)


if __name__ == "__main__":
    unittest.main()
