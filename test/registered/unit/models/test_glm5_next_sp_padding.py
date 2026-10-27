"""CPU regression for GLM DSA output rows at the input-scattered boundary.

Load the actual decoder forward without importing GPU-only model dependencies.
The boundary stub models the deferred-output wrapper and TP row divisibility;
GPU collective execution is covered by the PRO5000 integration experiment.
"""

import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch


class _OwedOutput:
    def __init__(self, tensor):
        self.tensor = tensor


class _MLP:
    def __call__(self, tensor, *args):
        return tensor


class _MoE:
    pass


def _decoder_forward(context):
    source = (
        Path(__file__).resolve().parents[4] / "python/sglang/srt/models/glm5_next.py"
    )
    tree = ast.parse(source.read_text())
    decoder = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Glm5NextDecoderLayer"
    )
    forward = next(
        node
        for node in decoder.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    forward.returns = None
    for arg in forward.args.args:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[forward], type_ignores=[]))
    namespace = {
        "torch": torch,
        "nullcontext": nullcontext,
        "get_attn_tp_context": lambda: context,
        "residual_access": SimpleNamespace(buffer=lambda tensor: tensor),
        "Glm5NextMLP": _MLP,
        "Glm5NextMoE": _MoE,
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["forward"]


class TestGlm5NextScatteredDsaPadding(unittest.TestCase):
    def run_decoder(
        self, logical_rows, *, returned_rows=None, scattered=True, kda=False
    ):
        context = SimpleNamespace(
            input_scattered=scattered, clear_attn_inputs=lambda: None
        )
        forward = _decoder_forward(context)
        physical_rows = (logical_rows + 7) // 8 * 8 if scattered else logical_rows
        if returned_rows is None:
            returned_rows = physical_rows if kda else logical_rows
        hidden = torch.arange(physical_rows * 4, dtype=torch.float32).reshape(-1, 4)
        topk_indices = object()

        def ffn_prepare(output, batch):
            # finish must wrap only after the padding has been restored.
            self.assertIsInstance(output, _OwedOutput)
            if scattered:
                self.assertEqual(output.tensor.shape[0] % 8, 0)
            return output.tensor

        decoder = SimpleNamespace(
            is_linear_attn=kda,
            attn_boundary=SimpleNamespace(
                prepare=lambda tensor, batch, **kwargs: tensor,
                input_on_attn_tp_slices=False,
                finish=lambda tensor, batch: _OwedOutput(tensor),
            ),
            self_attn=lambda **kwargs: (hidden[:returned_rows] * 2, topk_indices),
            ffn_boundary=SimpleNamespace(
                prepare=ffn_prepare, finish=lambda tensor, batch: tensor
            ),
            mlp=_MLP(),
        )
        batch = SimpleNamespace(
            input_ids=torch.zeros(physical_rows, dtype=torch.int64),
            global_num_token_non_padded_cpu=logical_rows,
        )
        output, indices = forward(decoder, None, hidden, batch)
        self.assertIs(indices, topk_indices)
        torch.testing.assert_close(output[:returned_rows], hidden[:returned_rows] * 2)
        return output, physical_rows

    def test_dsa_restores_collective_layout_and_preserves_real_rows(self):
        for rows in (6, 7, 9, 4103, 4096):
            with self.subTest(rows=rows):
                output, physical = self.run_decoder(rows)
                self.assertEqual(output.shape[0], physical)
                torch.testing.assert_close(
                    output[rows:], torch.zeros_like(output[rows:])
                )

    def test_sp_off_keeps_logical_rows(self):
        output, _ = self.run_decoder(6, scattered=False)
        self.assertEqual(output.shape[0], 6)

    def test_unexpected_loss_of_real_rows_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unexpected scattered attention rows"):
            self.run_decoder(6, returned_rows=5)

    def test_kda_physical_output_is_unchanged(self):
        output, physical = self.run_decoder(6, kda=True)
        self.assertEqual(output.shape[0], physical)


if __name__ == "__main__":
    unittest.main()
