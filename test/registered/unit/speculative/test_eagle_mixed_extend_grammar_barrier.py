"""Exercise the actual worker dispatch with CPU-only pending-result stubs.

The fake FSM exposes a stale grammar mask when an overlapping result has not
been committed before mixed-extend sampling. GPU kernels are not imported.
"""

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace


def _worker_forward():
    source = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/speculative/eagle_worker_v2.py"
    )
    tree = ast.parse(source.read_text())
    method = next(
        method
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for method in cls.body
        if isinstance(method, ast.FunctionDef)
        and method.name == "forward_batch_generation"
    )
    method = copy.deepcopy(method)
    method.decorator_list = []
    method.returns = None
    for node in ast.walk(method):
        if isinstance(node, ast.arg):
            node.annotation = None
    namespace = {}
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[method.name]


class _Grammar:
    def __init__(self):
        self.position = 0

    def allowed_token(self):
        return (10, 20, 30)[self.position]

    def accept_token(self, token):
        if token != self.allowed_token():
            raise ValueError(f"Token not accepted: {token}")
        self.position += 1


class TestEagleMixedExtendGrammarBarrier(unittest.TestCase):
    def run_dispatch(self, *, extend, mixed, grammar=True, callback=True, shared=True):
        events = []
        current = _Grammar()
        previous = current if shared else _Grammar()
        pending = callback

        def barrier():
            nonlocal pending
            if pending:
                previous.accept_token(10)
                pending = False
                events.append("commit_previous")

        def prefill(batch, on_publish, proxy):
            token = current.allowed_token() if batch.has_grammar else 99
            events.append(("sample", token))
            return token

        worker = SimpleNamespace(
            enable_dp_spec_prefill_coordination=False,
            _forward_prefill_batch=prefill,
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: extend),
            is_extend_in_batch=mixed,
            has_grammar=grammar,
        )
        token = _worker_forward()(
            worker, batch, grammar_barrier=barrier if callback else None
        )
        sampled_events = events[:]
        if callback:
            barrier()  # Scheduler may publish the preceding result afterwards.
        if grammar:
            current.accept_token(token)
        return token, sampled_events

    def test_mixed_extend_commits_before_sampling(self):
        token, events = self.run_dispatch(extend=True, mixed=True)
        self.assertEqual(token, 20)
        self.assertEqual(events, ["commit_previous", ("sample", 20)])

    def test_mixed_flag_without_extend_mode_also_waits(self):
        token, events = self.run_dispatch(extend=False, mixed=True)
        self.assertEqual(token, 20)
        self.assertEqual(events[0], "commit_previous")

    def test_new_prefill_preserves_its_own_grammar(self):
        token, events = self.run_dispatch(extend=True, mixed=False, shared=False)
        self.assertEqual(token, 10)
        self.assertEqual(events, ["commit_previous", ("sample", 10)])

    def test_no_grammar_does_not_wait_on_barrier(self):
        token, events = self.run_dispatch(extend=True, mixed=True, grammar=False)
        self.assertEqual(token, 99)
        self.assertEqual(events, [("sample", 99)])

    def test_mixed_without_callback_remains_supported(self):
        token, events = self.run_dispatch(extend=True, mixed=True, callback=False)
        self.assertEqual(token, 10)
        self.assertEqual(events, [("sample", 10)])

    def test_pure_prefill_without_callback_remains_supported(self):
        token, events = self.run_dispatch(
            extend=True, mixed=False, callback=False, shared=False
        )
        self.assertEqual(token, 10)
        self.assertEqual(events, [("sample", 10)])


if __name__ == "__main__":
    unittest.main()
