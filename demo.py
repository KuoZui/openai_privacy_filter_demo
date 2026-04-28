"""CLI benchmark for openai/privacy-filter (opf).

Measures:
  * model load time
  * process RSS delta after model load (and CUDA memory delta on GPU)
  * cold inference latency (first call)
  * warm inference latency (avg/min/max over N calls)

Inference uses the constrained Viterbi decoder by default.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psutil

DEFAULT_CHECKPOINT_DIR = Path.home() / ".opf" / "privacy_filter"

try:
    import torch
except ImportError as e:
    print("torch is required (installed via opf). Run: pip install -r requirements.txt", file=sys.stderr)
    raise

from opf import OPF


BUILTIN_SAMPLES: list[tuple[str, str]] = [
    ("en", "Hi, I'm Alice Chen. You can reach me at alice.chen@example.com or +1 (415) 555-0142."),
    ("en", "Please ship the package to 1600 Amphitheatre Pkwy, Mountain View, CA 94043 by 2024-12-31."),
    ("en", "Account 4242-4242-4242-4242 was charged; receipt at https://billing.example.com/r/abc123."),
    ("zh", "您好，我是林志明，手機 0912-345-678，住台北市信義區市府路 45 號。"),
    ("zh", "請於 2025/03/15 前匯款至帳號 700-1234567890，並 email 至 jiaming.lin@example.com.tw 確認。"),
    ("zh", "病患王小華（身分證 A123456789，生日 1988-07-22）下次回診請預約。"),
]


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2


def cuda_alloc_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def cuda_reserved_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024**2
    return 0.0


@dataclass
class Report:
    device: str
    decode_mode: str
    load_seconds: float
    memory: dict[str, float]
    inference: dict[str, Any]
    samples: list[dict[str, Any]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="inference device (default: cpu)")
    p.add_argument(
        "--decode-mode",
        choices=["viterbi", "argmax"],
        default="viterbi",
        help="span decoder (default: viterbi)",
    )
    p.add_argument("--warmup-runs", type=int, default=1, help="cold-run count before warm timing (default: 1)")
    p.add_argument("--measure-runs", type=int, default=5, help="warm-run sample count (default: 5)")
    p.add_argument("--text", default=None, help="redact a single custom text instead of built-in samples")
    p.add_argument("--json", dest="json_out", action="store_true", help="emit JSON-only output (no human report)")
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "model checkpoint directory; if missing, opf will auto-download to it. "
            f"defaults to $OPF_CHECKPOINT or {DEFAULT_CHECKPOINT_DIR}"
        ),
    )
    return p.parse_args()


def resolve_checkpoint_dir(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env = os.environ.get("OPF_CHECKPOINT")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_CHECKPOINT_DIR


def pick_samples(args: argparse.Namespace) -> list[str]:
    if args.text:
        return [args.text]
    return [text for _, text in BUILTIN_SAMPLES]


def cycle_n(items: list[str], n: int) -> list[str]:
    out = []
    for i in range(n):
        out.append(items[i % len(items)])
    return out


def build_baseline(device: str) -> tuple[float, float]:
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return rss_mb(), cuda_alloc_mb()


def load_model(device: str, decode_mode: str) -> tuple[OPF, float]:
    t0 = time.perf_counter()
    opf = OPF(device=device, decode_mode=decode_mode)
    # touch the runtime so any lazy weight materialization happens inside the timing window
    _ = opf.get_runtime()
    return opf, time.perf_counter() - t0


def time_inference(opf: OPF, texts: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, text in enumerate(texts):
        t0 = time.perf_counter()
        result = opf.redact(text)
        dt = time.perf_counter() - t0
        rows.append(
            {
                "index": idx,
                "seconds": dt,
                "input": text,
                "redacted_text": getattr(result, "redacted_text", None),
                "detected_spans": _serialize_spans(getattr(result, "detected_spans", None)),
            }
        )
    return rows


def _serialize_spans(spans: Any) -> list[dict[str, Any]]:
    if spans is None:
        return []
    out = []
    for s in spans:
        if hasattr(s, "to_dict"):
            out.append(s.to_dict())
        elif hasattr(s, "__dict__"):
            out.append({k: v for k, v in vars(s).items() if not k.startswith("_")})
        else:
            out.append({"repr": repr(s)})
    return out


def build_report(args: argparse.Namespace, samples: list[str]) -> Report:
    rss_before, cuda_before = build_baseline(args.device)

    opf, load_seconds = load_model(args.device, args.decode_mode)

    rss_after, cuda_after = rss_mb(), cuda_alloc_mb()
    cuda_reserved_after = cuda_reserved_mb()

    total_runs = max(1, args.warmup_runs + args.measure_runs)
    rows = time_inference(opf, cycle_n(samples, total_runs))

    cold_seconds = rows[0]["seconds"] if rows else 0.0
    warm_times = [r["seconds"] for r in rows[args.warmup_runs:]]
    inference: dict[str, Any] = {
        "cold_seconds": cold_seconds,
        "warm_avg_seconds": statistics.fmean(warm_times) if warm_times else None,
        "warm_min_seconds": min(warm_times) if warm_times else None,
        "warm_max_seconds": max(warm_times) if warm_times else None,
        "warm_samples": len(warm_times),
        "total_runs": len(rows),
    }

    memory = {
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": rss_after - rss_before,
    }
    if torch.cuda.is_available() and args.device == "cuda":
        memory["cuda_before_mb"] = cuda_before
        memory["cuda_after_mb"] = cuda_after
        memory["cuda_delta_mb"] = cuda_after - cuda_before
        memory["cuda_reserved_mb"] = cuda_reserved_after

    return Report(
        device=args.device,
        decode_mode=args.decode_mode,
        load_seconds=load_seconds,
        memory=memory,
        inference=inference,
        samples=rows,
    )


def fmt_mb(v: float) -> str:
    return f"{v:,.1f} MB"


def fmt_sec(v: float) -> str:
    return f"{v:.3f} s" if v >= 1 else f"{v * 1000:.1f} ms"


def print_human_report(rep: Report) -> None:
    print("=" * 68)
    print("OpenAI Privacy Filter — Benchmark Report")
    print("=" * 68)
    print(f"  device       : {rep.device}")
    print(f"  decode_mode  : {rep.decode_mode}")
    print(f"  load_time    : {fmt_sec(rep.load_seconds)}")
    print()
    print("Memory")
    print(f"  RSS before   : {fmt_mb(rep.memory['rss_before_mb'])}")
    print(f"  RSS after    : {fmt_mb(rep.memory['rss_after_mb'])}")
    print(f"  RSS delta    : {fmt_mb(rep.memory['rss_delta_mb'])}   <-- model footprint on host")
    if "cuda_delta_mb" in rep.memory:
        print(f"  CUDA delta   : {fmt_mb(rep.memory['cuda_delta_mb'])}   (allocated)")
        print(f"  CUDA reserved: {fmt_mb(rep.memory['cuda_reserved_mb'])} (cached pool)")
    print()
    print("Inference")
    print(f"  cold (1st)   : {fmt_sec(rep.inference['cold_seconds'])}")
    if rep.inference["warm_avg_seconds"] is not None:
        print(
            f"  warm avg     : {fmt_sec(rep.inference['warm_avg_seconds'])}"
            f"   (min {fmt_sec(rep.inference['warm_min_seconds'])},"
            f" max {fmt_sec(rep.inference['warm_max_seconds'])},"
            f" n={rep.inference['warm_samples']})"
        )
    print()
    print("Redactions")
    print("-" * 68)
    for row in rep.samples:
        tag = "cold" if row["index"] == 0 else "warm"
        print(f"[{tag} #{row['index']}] {fmt_sec(row['seconds'])}")
        print(f"  input    : {row['input']}")
        print(f"  redacted : {row['redacted_text']}")
        if row["detected_spans"]:
            print(f"  spans    : {len(row['detected_spans'])} found")
            for s in row["detected_spans"]:
                print(f"             - {s}")
        else:
            print("  spans    : (none)")
        print()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: --device cuda was requested but torch.cuda.is_available() is False.", file=sys.stderr)
        return 2

    # Only force a checkpoint path if the user explicitly asked for one.
    # Leaving OPF_CHECKPOINT unset lets `opf` auto-download to ~/.opf/privacy_filter
    # via huggingface_hub when the default location is missing.
    if args.checkpoint_dir:
        os.environ["OPF_CHECKPOINT"] = str(Path(args.checkpoint_dir).expanduser().resolve())
    ckpt = resolve_checkpoint_dir(args.checkpoint_dir)
    if not args.json_out:
        exists = ckpt.exists() and any(ckpt.iterdir())
        status = "found" if exists else "will auto-download from Hugging Face on first load"
        print(f"[checkpoint] {ckpt} ({status})")

    samples = pick_samples(args)
    report = build_report(args, samples)

    if args.json_out:
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    else:
        print_human_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
