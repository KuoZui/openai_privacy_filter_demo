"""Streamlit Web UI for openai/privacy-filter (opf).

Launch:
    streamlit run app.py
    streamlit run app.py -- --device cuda
"""

from __future__ import annotations

import os

# Cap PyTorch / OpenMP threads BEFORE importing torch.
# Rationale: on shared-CPU hosts (Railway, etc.) PyTorch's default of "use all
# cores it sees" causes thread contention rather than speedup. Empirically 4
# threads is a sweet spot for this 1.5B model on shared vCPUs.
# Override via OPF_NUM_THREADS env var if needed.
_NUM_THREADS = os.environ.get("OPF_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", _NUM_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _NUM_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NUM_THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _NUM_THREADS)

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import streamlit as st

DEFAULT_CHECKPOINT_DIR = Path.home() / ".opf" / "privacy_filter"

try:
    import torch
except ImportError:
    st.error("torch is required (installed via opf). Run: pip install -r requirements.txt")
    st.stop()

# Apply thread limits at the torch level too (env vars alone don't always stick
# once torch is loaded). Safe to call multiple times.
try:
    torch.set_num_threads(int(_NUM_THREADS))
    torch.set_num_interop_threads(1)
except RuntimeError:
    # set_num_interop_threads must be called before any parallel work; if Streamlit
    # already kicked off some torch ops on a previous rerun, this raises — ignore.
    pass

from opf import OPF


SAMPLE_EN = (
    "Hi, I'm Alice Chen — alice.chen@example.com or +1 (415) 555-0142. "
    "Please ship to 1600 Amphitheatre Pkwy, Mountain View, CA 94043 by 2024-12-31."
    "\n\n"
    "John Doe, born on 05/15/1980, lives at 123 Main Street, Anytown, CA 90210. His phone number is (555) 123-4567 and his email is john.doe@example.com. He works for Example Corp."
)
SAMPLE_ZH = (
    "您好，我是林志明，手機 0912-345-678，住台北市信義區市府路 45 號。"
    "請於 2025/03/15 前匯款至帳號 700-1234567890，並 email 至 jiaming.lin@example.com.tw 確認。"
)


def parse_cli_args() -> argparse.Namespace:
    # streamlit forwards args after `--` to sys.argv
    argv = sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--decode-mode", choices=["viterbi", "argmax"], default="viterbi")
    p.add_argument("--checkpoint-dir", default=None)
    return p.parse_args(argv)


def resolve_checkpoint_dir(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env = os.environ.get("OPF_CHECKPOINT")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_CHECKPOINT_DIR


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2


def cuda_alloc_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


@st.cache_resource(show_spinner="Loading openai/privacy-filter model (first run downloads ~3 GB)...")
def load_opf(device: str) -> tuple[OPF, dict[str, float]]:
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    rss0 = rss_mb()
    cuda0 = cuda_alloc_mb()
    t0 = time.perf_counter()
    # Always load with viterbi; decode_mode can be overridden per-call via set_decode_mode
    opf = OPF(device=device, decode_mode="viterbi")
    _ = opf.get_runtime()
    metrics = {
        "load_seconds": time.perf_counter() - t0,
        "rss_before_mb": rss0,
        "rss_after_mb": rss_mb(),
        "rss_delta_mb": rss_mb() - rss0,
        "cuda_delta_mb": cuda_alloc_mb() - cuda0,
    }
    return opf, metrics


def spans_to_rows(spans: Any, source_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not spans:
        return rows
    for s in spans:
        d = s.to_dict() if hasattr(s, "to_dict") else (vars(s) if hasattr(s, "__dict__") else {})
        start = d.get("start") or d.get("start_offset") or d.get("begin")
        end = d.get("end") or d.get("end_offset") or d.get("stop")
        ptype = d.get("type") or d.get("label") or d.get("category") or "?"
        original = d.get("text")
        if original is None and isinstance(start, int) and isinstance(end, int):
            original = source_text[start:end]
        rows.append({"type": ptype, "start": start, "end": end, "original_text": original})
    return rows


def main() -> None:
    args = parse_cli_args()
    # Only force OPF_CHECKPOINT when user explicitly passed --checkpoint-dir;
    # otherwise leave it unset so `opf` will auto-download to ~/.opf/privacy_filter.
    if args.checkpoint_dir:
        os.environ["OPF_CHECKPOINT"] = str(Path(args.checkpoint_dir).expanduser().resolve())
    ckpt = resolve_checkpoint_dir(args.checkpoint_dir)

    st.set_page_config(page_title="OpenAI Privacy Filter Demo", layout="wide")
    st.title("OpenAI Privacy Filter — Demo")
    st.caption(
        "互動式展示 [openai/privacy-filter](https://github.com/openai/privacy-filter)"
        " 的 PII 偵測與遮罩功能。預設使用 Viterbi 解碼。"
    )

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Settings")
        st.write(f"**Device**: `{args.device}`")
        ckpt_exists = ckpt.exists() and any(ckpt.iterdir()) if ckpt.exists() else False
        st.write(
            f"**Checkpoint**: `{ckpt}` "
            f"({'found' if ckpt_exists else 'will download on first load'})"
        )
        if args.device == "cuda" and not torch.cuda.is_available():
            st.error("--device cuda 但 torch.cuda.is_available() 為 False")
            st.stop()

        decode_mode = st.selectbox(
            "Decode mode",
            options=["viterbi", "argmax"],
            index=0 if args.decode_mode == "viterbi" else 1,
            help="viterbi: constrained BIOES path. argmax: independent per-token argmax.",
        )

        if st.button("重新載入模型", help="清除快取並重新載入（用於切換 device 後）"):
            load_opf.clear()
            st.rerun()

        st.divider()
        st.caption("啟動方式：`streamlit run app.py -- --device cuda`")

    # ---- Load model ----
    opf, metrics = load_opf(args.device)
    opf.set_decode_mode(decode_mode)

    # ---- Top metrics ----
    cols = st.columns(4)
    cols[0].metric("Load time", f"{metrics['load_seconds']:.2f} s")
    cols[1].metric("RSS delta", f"{metrics['rss_delta_mb']:,.0f} MB")
    if args.device == "cuda":
        cols[2].metric("CUDA delta", f"{metrics['cuda_delta_mb']:,.0f} MB")
    else:
        cols[2].metric("CUDA delta", "N/A")
    cols[3].metric("Decode mode", decode_mode)

    st.divider()

    # ---- Input area ----
    if "input_text" not in st.session_state:
        st.session_state.input_text = SAMPLE_EN

    btn_cols = st.columns([1, 1, 6])
    if btn_cols[0].button("載入英文範例"):
        st.session_state.input_text = SAMPLE_EN
        st.rerun()
    if btn_cols[1].button("載入中文範例"):
        st.session_state.input_text = SAMPLE_ZH
        st.rerun()

    text = st.text_area(
        "輸入文字（可包含 email / 姓名 / 電話 / 地址 / 日期 等 PII）",
        key="input_text",
        height=160,
    )

    run = st.button("Redact", type="primary", width="content")

    # ---- Output ----
    if run:
        if not text.strip():
            st.warning("請先輸入文字")
            return

        t0 = time.perf_counter()
        result = opf.redact(text)
        elapsed = time.perf_counter() - t0

        st.success(f"推論耗時 {elapsed * 1000:.1f} ms（decode_mode = {decode_mode}）")

        out_left, out_right = st.columns(2)
        with out_left:
            st.subheader("Redacted text")
            st.code(getattr(result, "redacted_text", str(result)), language=None, wrap_lines=True)
        with out_right:
            st.subheader("Detected spans")
            rows = spans_to_rows(getattr(result, "detected_spans", None), text)
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True)
            else:
                st.info("（沒有偵測到 PII）")

        with st.expander("RedactionResult (raw)"):
            if hasattr(result, "to_dict"):
                st.json(result.to_dict())
            else:
                st.write(result)


if __name__ == "__main__":
    main()
