"""Compare cold-cache KDA requests against a separately launched server variant."""

import argparse
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-path", default="/model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--mode", choices=("perf", "quality", "profile"), default="perf"
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[130048])
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--profile-flag", type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    rows = []

    def post(path, body):
        data = json.dumps(body).encode()
        try:
            return urllib.request.urlopen(
                urllib.request.Request(
                    args.base_url + path,
                    data=data,
                    headers={"Content-Type": "application/json"},
                ),
                timeout=1200,
            )
        except urllib.error.HTTPError as error:
            raise RuntimeError(error.read().decode()) from error

    def prompt_ids(n, seed):
        prefix = tok.encode(
            f"Benchmark {seed}. Read the following technical notes. ",
            add_special_tokens=False,
        )
        fill = tok.encode(
            "The system processes records in order. Each record describes a sensor "
            "reading, an observation, and a verification step. Stable service "
            "requires accurate inputs and consistent output. ",
            add_special_tokens=False,
        )
        end = tok.encode(
            " Summarize the technical notes in detail, using complete sentences.",
            add_special_tokens=False,
        )
        ids = prefix + (fill * (n // len(fill) + 2))[: n - len(prefix) - len(end)] + end
        assert len(ids) == n
        return ids

    def measure(n, repeat, *, warmup=False, output=128):
        ids = prompt_ids(n, f"kda-host-length-{n}-{repeat}")
        body = {
            "input_ids": ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output,
                "ignore_eos": True,
                "sampling_seed": 1234,
            },
            "stream": True,
            "cache_salt": args.phase + "-" + uuid.uuid4().hex,
        }
        # Serialize before starting the timer, including for the 1M request.
        request = urllib.request.Request(
            args.base_url + "/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        print("START", args.phase, n, repeat, "warmup", warmup, flush=True)
        t0 = time.perf_counter()
        first = last = None
        first_count = count = 0
        done = False
        events = []
        output_ids = []
        final = {}
        with urllib.request.urlopen(request, timeout=1200) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                final = json.loads(data)
                now = time.perf_counter()
                current = final["meta_info"].get("completion_tokens", 0)
                chunk = final.get("output_ids", [])
                if len(chunk) == current:
                    output_ids = chunk
                elif len(chunk) == current - count:
                    output_ids.extend(chunk)
                if current > count:
                    if first is None:
                        first, first_count = now, current
                    last, count = now, current
                    events.append([now - t0, count])
        elapsed = time.perf_counter() - t0
        meta = final["meta_info"]
        assert done and first is not None and count == len(output_ids) == output
        assert meta["prompt_tokens"] == n and meta.get("cached_tokens", 0) == 0
        assert first_count == 1
        row = {
            "phase": args.phase,
            "kind": "warmup" if warmup else "performance",
            "input_tokens": n,
            "repeat": repeat,
            "ttft_s": first - t0,
            "decode_tps": (count - first_count) / (last - first),
            "e2e_s": elapsed,
            "output_tokens": count,
            "cached_tokens": meta.get("cached_tokens", 0),
            "input_sha256": digest(ids),
            "output_sha256": digest(output_ids),
        }
        (root / f"perf-{n}-{repeat}-{warmup}.json").write_text(
            json.dumps({"result": row, "events": events, "final": final}, indent=2)
        )
        with (root / "results.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        rows.append(row)
        print("RESULT", json.dumps(row), flush=True)

    def quality():
        for n in (4096, 130048):
            for repetition in range(3):
                suffix = f"-repeat{repetition}" if repetition else ""
                ids = prompt_ids(n, "kda-logprob-parity")
                with post(
                    "/generate",
                    {
                        "input_ids": ids,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                        "return_logprob": True,
                        "logprob_start_len": n - 32,
                        "top_logprobs_num": 5,
                        "cache_salt": uuid.uuid4().hex,
                    },
                ) as response:
                    result = json.load(response)
                assert result["meta_info"].get("cached_tokens", 0) == 0
                (root / f"logprobs-{n}{suffix}.json").write_text(
                    json.dumps(result, indent=2)
                )
        code = "KDAHOSTLENGTHCHECK7319"
        filler = "Background note: the system stores routine sensor records.\n"

        def messages(k):
            return [
                {
                    "role": "user",
                    "content": "Read the document and find its unique verification code.\n"
                    + filler * k
                    + f"\nThe unique verification code is {code}.\n"
                    + filler * k
                    + "\nReply only with the verification code.",
                }
            ]

        lo, hi = 0, 130048 // (2 * len(tok.encode(filler))) + 8
        while lo < hi:
            mid = (lo + hi + 1) // 2
            ids = tok.apply_chat_template(messages(mid), tokenize=True, thinking=False)
            if len(ids) <= 130048:
                lo = mid
            else:
                hi = mid - 1
        with post(
            "/v1/chat/completions",
            {
                "model": "glm-5.3-flash",
                "messages": messages(lo),
                "temperature": 0,
                "max_tokens": 512,
                "chat_template_kwargs": {"thinking": False},
            },
        ) as response:
            result = json.load(response)
        (root / "needle.json").write_text(json.dumps(result, indent=2))
        message = result["choices"][0]["message"]
        assert code in (message.get("content") or "")
        print("QUALITY_PASSED", args.phase, flush=True)

    def profile():
        assert args.progress_file and args.profile_flag
        args.progress_file.touch()
        args.profile_flag.touch()
        armed = set()

        def arm(suffix):
            directory = root / ("trace-" + suffix)
            directory.mkdir()
            with post(
                "/start_profile",
                {
                    "output_dir": str(directory),
                    "num_steps": 3,
                    "activities": ["CPU", "GPU"],
                    "with_stack": False,
                    "record_shapes": False,
                    "merge_profiles": False,
                    "detailed_annotations": False,
                    "profile_prefix": args.phase + "-" + suffix,
                },
            ) as response:
                response.read()
            armed.add(suffix)
            print("PROFILE_ARM", args.phase, suffix, flush=True)

        errors = []

        def work():
            try:
                measure(130048, 99)
            except BaseException as error:
                errors.append(error)

        try:
            with args.progress_file.open() as f:
                f.seek(0, 2)
                arm("early")
                thread = threading.Thread(target=work)
                thread.start()
                while thread.is_alive():
                    for line in f.readlines():
                        row = json.loads(line)
                        if row["mode"] == "EXTEND":
                            if row["kv_end"] >= 130048 // 2 and "mid" not in armed:
                                arm("mid")
                            elif row["kv_end"] >= 130048 - 8192 and "late" not in armed:
                                arm("late")
                        elif row["mode"] == "DECODE" and "decode" not in armed:
                            arm("decode")
                    time.sleep(0.02)
                thread.join()
            if errors:
                raise errors[0]
            assert armed == {"early", "mid", "late", "decode"}
            assert len(list(root.rglob("*.trace.json.gz"))) == 32
        finally:
            args.profile_flag.unlink(missing_ok=True)

    if args.mode == "quality":
        quality()
    elif args.mode == "profile":
        profile()
    else:
        if args.warmup:
            for n in (2048, 31744, 130048):
                measure(n, 0, warmup=True, output=32)
        for n in args.lengths:
            measure(n, args.repeat)
    (root / "final.json").write_text(json.dumps({"status": "passed", "rows": rows}))


if __name__ == "__main__":
    main()
