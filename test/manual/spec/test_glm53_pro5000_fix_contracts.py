"""CPU control-flow regressions for the PRO5000 GLM serving fixes.

Extract the actual methods to avoid importing CUDA-only serving dependencies.
These tests validate configuration lifetimes and ordering, not GPU numerics.
Run directly with Python's standard library; no model or GPU is required.
"""

import ast
import contextlib
import copy
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"
WORKER = SRT / "speculative/eagle_worker_v2.py"


def load_function(path, name, namespace, *, owner=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if owner is not None:
        nodes = next(
            n.body for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner
        )
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node = copy.deepcopy(node)
    node.returns = None
    for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class TestAdaptiveCaptureConfig(unittest.TestCase):
    def check_override(self, batches, raise_inside):
        original = SimpleNamespace(
            decode=SimpleNamespace(bs=[1, 2, 4, 8, 16, 32], max_bs=32, backend="full"),
            prefill=SimpleNamespace(backend="disabled"),
        )
        graph = SimpleNamespace(
            cuda_graph_config=original,
            cuda_graph_bs_decode=[1, 2, 4, 8, 16, 32],
            disable_cuda_graph=False,
        )
        spec = SimpleNamespace(speculative_num_steps=3, speculative_num_draft_tokens=4)
        runner = SimpleNamespace(attn_backend=object(), draft_attn_backend=object())
        draft = SimpleNamespace(
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            draft_attn_backend=object(),
            draft_extend_attn_backend=object(),
            draft_runner=runner,
            cuda_graph_runner=object(),
            cuda_graph_runner_for_draft_extend=object(),
            _rebuild_topk1_chain_buffers=Mock(),
        )
        worker = SimpleNamespace(
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            _draft_worker=draft,
        )

        def override(_reason, **changes):
            for key, value in changes.items():
                setattr(spec if key.startswith("speculative_") else graph, key, value)

        fn = load_function(
            WORKER,
            "_override_worker_state",
            {
                "contextlib": contextlib,
                "get_exec": lambda: SimpleNamespace(graph=graph),
                "get_spec": lambda: spec,
                "get_context": lambda: SimpleNamespace(override=override),
                "Backend": SimpleNamespace(DISABLED="disabled"),
            },
            owner="EAGLEWorkerV2",
        )
        old_legacy = graph.cuda_graph_bs_decode
        old_runner_backend = runner.attn_backend

        def capture():
            with fn(worker, 1, 2, cuda_graph_bs=batches):
                self.assertEqual(worker.speculative_num_steps, 1)
                self.assertEqual(spec.speculative_num_draft_tokens, 2)
                if batches is None:
                    self.assertIs(graph.cuda_graph_config, original)
                    self.assertIs(graph.cuda_graph_bs_decode, old_legacy)
                else:
                    resolved = graph.cuda_graph_config
                    self.assertIsNot(resolved, original)
                    self.assertEqual(resolved.decode.bs, batches)
                    self.assertEqual(resolved.decode.max_bs, max(batches, default=1))
                    self.assertEqual(graph.cuda_graph_bs_decode, batches)
                    self.assertEqual(graph.disable_cuda_graph, not bool(batches))
                    self.assertEqual(
                        resolved.decode.backend, "full" if batches else "disabled"
                    )
                    self.assertEqual(original.decode.bs, [1, 2, 4, 8, 16, 32])
                    # Capturing one state must not mutate the caller's bucket list.
                    resolved.decode.bs.append(64)
                    self.assertNotIn(64, batches)
                runner.attn_backend = "capture-only"
                if raise_inside:
                    raise RuntimeError("capture failed")

        if raise_inside:
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                capture()
        else:
            capture()
        self.assertIs(graph.cuda_graph_config, original)
        self.assertIs(graph.cuda_graph_bs_decode, old_legacy)
        self.assertFalse(graph.disable_cuda_graph)
        self.assertEqual(original.decode.backend, "full")
        self.assertEqual(worker.speculative_num_steps, 3)
        self.assertEqual(draft.speculative_num_draft_tokens, 4)
        self.assertEqual(spec.speculative_num_steps, 3)
        self.assertIs(runner.attn_backend, old_runner_backend)
        self.assertEqual(draft._rebuild_topk1_chain_buffers.call_count, 2)

    def test_capture_and_restore(self):
        for batches in (None, [], [1], [1, 2, 4, 8, 16, 32]):
            for raise_inside in (False, True):
                with self.subTest(batches=batches, raise_inside=raise_inside):
                    self.check_override(batches, raise_inside)


class TestGrammarBarrierOrder(unittest.TestCase):
    def test_barrier_precedes_target_forward_for_extend_and_mixed(self):
        fn = load_function(
            WORKER,
            "forward_batch_generation",
            {"CaptureHiddenMode": SimpleNamespace(NULL=0, FULL=1)},
            owner="EAGLEWorkerV2",
        )
        for extend in (True, False):
            for constrained in (True, False):
                with self.subTest(extend=extend, constrained=constrained):
                    events = []
                    grammar_position = [0]

                    def barrier():
                        events.append("previous-result")
                        grammar_position[0] = 1

                    def target(*_args, **_kwargs):
                        events.append("target-forward")
                        # A pending grammar result must be committed before mask creation.
                        self.assertEqual(grammar_position[0], int(constrained))
                        return SimpleNamespace()

                    batch = SimpleNamespace(
                        forward_mode=SimpleNamespace(is_extend=lambda: extend),
                        is_extend_in_batch=not extend,
                        has_grammar=constrained,
                        seq_lens=[8],
                    )
                    worker = SimpleNamespace(
                        speculative_algorithm=SimpleNamespace(
                            is_standalone=lambda: False
                        ),
                        target_worker=SimpleNamespace(forward_batch_generation=target),
                        _draft_worker=None,
                    )
                    fn(worker, batch, grammar_barrier=barrier)
                    expected = (
                        ["previous-result", "target-forward"]
                        if constrained
                        else ["target-forward"]
                    )
                    self.assertEqual(events, expected)


class TestHiCacheGuards(unittest.TestCase):
    def test_packed_rows_require_matching_width_and_storage_dtype(self):
        class MLA:
            pass

        class Hybrid:
            pass

        fake = ModuleType("sglang.srt.mem_cache.memory_pool")
        fake.MLATokenToKVPool = MLA
        fake.HybridLinearKVPool = Hybrid
        with patch.dict(sys.modules, {fake.__name__: fake}):
            fn = load_function(
                SRT / "speculative/base_spec_worker.py", "_hicache_mtp_rows_match", {}
            )
            target, draft = MLA(), MLA()
            target.kv_cache_dim = draft.kv_cache_dim = 512
            target.store_dtype = draft.store_dtype = "bfloat16"
            self.assertTrue(fn(target, [draft]))
            draft.kv_cache_dim = 576
            self.assertFalse(fn(target, [draft]))
            draft.kv_cache_dim = 512
            draft.store_dtype = "float8"
            self.assertFalse(fn(target, [draft]))
            wrapped_target, wrapped_draft = Hybrid(), Hybrid()
            wrapped_target.full_kv_pool, wrapped_draft.full_kv_pool = target, draft
            self.assertFalse(fn(wrapped_target, [wrapped_draft]))
            draft.store_dtype = "bfloat16"
            self.assertTrue(fn(wrapped_target, [wrapped_draft]))

    def test_finished_checkpoint_cannot_exceed_prefix_key(self):
        fn = load_function(
            SRT / "mem_cache/unified_cache/components/mamba.py",
            "prepare_for_caching_req",
            {},
            owner="MambaComponent",
        )
        component = SimpleNamespace(
            cache=SimpleNamespace(enable_mamba_extra_buffer=True)
        )
        req = SimpleNamespace(kv=SimpleNamespace(mamba_last_track_seqlen=1024))
        self.assertEqual(fn(component, req, SimpleNamespace(), 1000, True), 0)


if __name__ == "__main__":
    unittest.main()
