#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Paper-oriented PDF to Markdown OCR."""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageOps
import pdf2image
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI


OCR_MD_PROMPT = "Extract all text and math from the image and output Markdown only."
OCR_MD_PROMPT_FALLBACK = "OCR to Markdown. Output only the content."


# ---------------- config helpers ----------------

def find_config_json() -> Path:
    p = Path.cwd() / "config.json"
    if p.exists():
        return p.resolve()
    here = Path(__file__).resolve().parent
    for d in [here] + list(here.parents):
        q = d / "config.json"
        if q.exists():
            return q.resolve()
    raise FileNotFoundError("config.json not found (checked CWD and script parents).")


def load_config() -> Dict:
    cfg_path = find_config_json()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{cfg_path} must contain a JSON object.")
    return data


def require_str(cfg: Dict, key: str) -> str:
    v = cfg.get(key)
    if not isinstance(v, str) or not v.strip():
        raise KeyError(f"Missing/invalid '{key}' in config.json")
    return v.strip()


def get_cfg(cfg: Dict, key: str, default):
    return cfg.get(key, default)


# ---------------- settings helpers (module-local JSON) ----------------

def find_settings_json() -> Path:
    p = Path(__file__).resolve().with_name("settings.json")
    if p.exists():
        return p.resolve()
    raise FileNotFoundError(f"settings.json not found next to script: {p}")


def load_settings() -> Dict[str, Any]:
    sp = find_settings_json()
    data = json.loads(sp.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{sp} must contain a JSON object.")
    return data


def get_setting(settings: Dict[str, Any], name: str, default: Any) -> Any:
    return settings.get(name, default)


# ---------------- PDF render ----------------

def render_pdf_to_pil_images(pdf_path: Path, dpi: int) -> List[Image.Image]:
    images = pdf2image.convert_from_path(
        str(pdf_path),
        dpi=dpi,
        fmt="png",
        thread_count=2,
    )
    out: List[Image.Image] = []
    for im in images:
        if im.mode != "RGB":
            im = im.convert("RGB")
        out.append(im)
    return out


def render_single_page(pdf_path: Path, dpi: int, page_1based: int) -> Image.Image:
    images = pdf2image.convert_from_path(
        str(pdf_path),
        dpi=dpi,
        fmt="png",
        first_page=page_1based,
        last_page=page_1based,
        thread_count=1,
    )
    if not images:
        raise RuntimeError(f"Failed to render page {page_1based} at dpi={dpi}")
    im = images[0]
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


# ---------------- image utilities ----------------

def pad_image(img: Image.Image, pad_top: int = 0, pad_other: int = 0) -> Image.Image:
    """Add white padding (helps OCR not miss edge-touching glyphs)."""
    if pad_top <= 0 and pad_other <= 0:
        return img
    border = (pad_other, pad_top, pad_other, pad_other)  # left, top, right, bottom
    return ImageOps.expand(img, border=border, fill="white")


def split_two_columns(img: Image.Image, overlap: int = 30) -> Tuple[Image.Image, Image.Image]:
    """Split image into left/right halves with small overlap."""
    w, h = img.size
    mid = w // 2
    ov = max(0, int(overlap))
    left = img.crop((0, 0, min(w, mid + ov), h))
    right = img.crop((max(0, mid - ov), 0, w, h))
    return left, right


def save_debug_images(
    debug_dir: Path,
    page_i: int,
    raw_img: Image.Image,
    ocr_img: Image.Image,
    tag: str,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    try:
        raw_img.save(debug_dir / f"page_{page_i:04d}_{tag}_raw.png", "PNG", optimize=True)
        ocr_img.save(debug_dir / f"page_{page_i:04d}_{tag}_ocr.png", "PNG", optimize=True)
    except Exception:
        pass


def save_debug_column_images(
    debug_dir: Path,
    page_i: int,
    left_img: Image.Image,
    right_img: Image.Image,
    tag: str,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    try:
        left_img.save(debug_dir / f"page_{page_i:04d}_{tag}_col_left.png", "PNG", optimize=True)
        right_img.save(debug_dir / f"page_{page_i:04d}_{tag}_col_right.png", "PNG", optimize=True)
    except Exception:
        pass


# ---------------- OCR helpers ----------------

def pil_image_to_data_url(img: Image.Image) -> str:
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^\s*```(?:md|markdown)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


_LAYOUT_LINE_RE = re.compile(
    r"^\s*(text|equation|interline_equation|sub_title|title|figure|table)\s*\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]",
    re.IGNORECASE
)

def looks_like_layout(md: str) -> bool:
    """Only return True if enough layout-tag lines appear in the first chunk."""
    s = (md or "").strip()
    if not s:
        return True
    lines = [ln for ln in s.splitlines()[:80] if ln.strip()]
    if not lines:
        return True
    tag_cnt = sum(1 for ln in lines if _LAYOUT_LINE_RE.search(ln))
    return tag_cnt >= 8 and (tag_cnt / max(1, len(lines))) >= 0.25


def strip_layout_boxes(md: str) -> str:
    lines = (md or "").splitlines()
    out = []
    for ln in lines:
        ln2 = re.sub(r"^\s*\w+\s*\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]\s*", "", ln)
        if ln2.strip():
            out.append(ln2.rstrip())
    return "\n".join(out).strip()


_DO_NOT_CHANGE_RE = re.compile(
    r"^\s*do\s+not\s+change\s+the\s+text(\s+in\s+the\s+image)?\s*\.?\s*$",
    re.IGNORECASE
)

def strip_boilerplate(md: str) -> str:
    lines = (md or "").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    removed = 0
    while lines and removed < 3 and _DO_NOT_CHANGE_RE.match(lines[0]):
        lines.pop(0)
        removed += 1
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip()


def looks_bad(md: str) -> bool:
    s = (md or "").strip()
    if not s:
        return True
    head = s[:800].lower()
    if "sub-sub-sub" in head:
        return True
    echo_phrases = [
        "ocr to markdown",
        "output markdown only",
        "output only the content",
        "extract all text and math",
    ]
    if len(s) < 120 and any(head.startswith(p) for p in echo_phrases):
        return True
    return False


def is_likely_two_column_image(img: Image.Image) -> bool:
    """Conservative two-column detector."""
    import numpy as np

    w, h = img.size
    target_w = 900
    if w > target_w:
        scale = target_w / float(w)
        img2 = img.resize((target_w, max(1, int(h * scale))), Image.BILINEAR)
    else:
        img2 = img

    g = img2.convert("L")
    a = np.array(g, dtype=np.uint8)

    ink = a < 200

    H, W = ink.shape
    if W < 200 or H < 200:
        return False

    y0 = int(H * 0.12)
    y1 = int(H * 0.88)
    core = ink[y0:y1, :]

    mid = W // 2
    band = max(10, int(W * 0.06))
    side = max(10, int(W * 0.18))

    cx0, cx1 = max(0, mid - band), min(W, mid + band)
    lx0, lx1 = 0, min(W, side)
    rx0, rx1 = max(0, W - side), W

    central_ink_ratio = core[:, cx0:cx1].mean()
    left_ink_ratio = core[:, lx0:lx1].mean()
    right_ink_ratio = core[:, rx0:rx1].mean()

    if central_ink_ratio > 0.015:
        return False
    if left_ink_ratio < 0.035 or right_ink_ratio < 0.035:
        return False

    row_central = core[:, cx0:cx1].mean(axis=1)
    empty_rows = (row_central < 0.01).mean()
    if empty_rows < 0.72:
        return False

    return True


def looks_like_interleaved_columns(md: str) -> bool:
    s = (md or "").strip()
    if len(s) < 400:
        return False
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if len(lines) < 40:
        return False

    short = sum(1 for ln in lines if len(ln) <= 35)
    very_short = sum(1 for ln in lines if len(ln) <= 18)
    avg_len = sum(len(ln) for ln in lines) / max(1, len(lines))

    if (short / len(lines)) >= 0.55 and avg_len <= 55:
        return True
    if (very_short / len(lines)) >= 0.30 and avg_len <= 65:
        return True
    return False


# ---------------- General post-processing (page-level) ----------------

_EQNUM_RE = re.compile(r"^\(\s*\d+(?:\.\d+)*\s*\)$")
_DOLLAR_FENCE_RE = re.compile(r"^\s*\$\$\s*$")

def _basic_brace_balance_score(s: str) -> int:
    # Very cheap: only counts (), {}, []
    pairs = {"(": ")", "{": "}", "[": "]"}
    opens = set(pairs.keys())
    closes = set(pairs.values())
    stack: List[str] = []
    bad = 0
    for ch in s:
        if ch in opens:
            stack.append(ch)
        elif ch in closes:
            if not stack:
                bad += 1
            else:
                op = stack.pop()
                if pairs.get(op) != ch:
                    bad += 1
    bad += len(stack)
    return bad  # 0 is best

def count_unbalanced_display_math(md: str) -> int:
    # counts $$ fences; odd -> unbalanced
    lines = (md or "").splitlines()
    fence = sum(1 for ln in lines if _DOLLAR_FENCE_RE.match(ln))
    return fence % 2  # 0 ok, 1 bad

def bind_equation_numbers(md: str) -> Tuple[str, Dict[str, Any]]:
    """
    Generic equation-number binding:
    If a standalone line "(1.1)" appears right after a $$...$$ block, move it into the block via \tag{1.1}.
    """
    lines = (md or "").splitlines()
    if not lines:
        return md, {"eqnum_moved": 0}

    moved = 0
    out: List[str] = []
    i = 0

    # Keep a small buffer of recent output indices for "last $$ block"
    while i < len(lines):
        ln = lines[i]
        if _EQNUM_RE.match(ln.strip()):
            # Find last display-math block in out that has a closing $$ we can inject before.
            tag = ln.strip()[1:-1].strip()
            j = len(out) - 1

            # Skip trailing blank lines in out
            while j >= 0 and not out[j].strip():
                j -= 1

            # We want the most recent closing $$ fence
            if j >= 0 and _DOLLAR_FENCE_RE.match(out[j]):
                # Find matching opening $$ fence (scan backward)
                k = j - 1
                while k >= 0 and not _DOLLAR_FENCE_RE.match(out[k]):
                    k -= 1
                if k >= 0:
                    # Inject \tag{...} just before closing fence, if no \tag already present.
                    has_tag = False
                    for t in range(k + 1, j):
                        if r"\tag{" in out[t]:
                            has_tag = True
                            break
                    if not has_tag:
                        out.insert(j, rf"\tag{{{tag}}}")
                        moved += 1
                        i += 1
                        # Also swallow a few trailing blank lines after eqnum
                        while i < len(lines) and not lines[i].strip():
                            i += 1
                        continue
            # If we cannot bind, keep the line as-is
            out.append(ln)
            i += 1
            continue

        out.append(ln)
        i += 1

    return "\n".join(out).strip(), {"eqnum_moved": moved}

def normalize_whitespace(md: str) -> str:
    s = (md or "").replace("\u00a0", " ")
    s = re.sub(r"\n{4,}", "\n\n\n", s)
    return s.strip()

def postprocess_page_markdown(md: str) -> Tuple[str, Dict[str, Any]]:
    """
    General, non-paper-specific cleanup.
    """
    meta: Dict[str, Any] = {}
    s = (md or "").strip()
    s = normalize_whitespace(s)

    s2, m2 = bind_equation_numbers(s)
    meta.update(m2)
    s = s2

    # Keep a couple simple signals for scoring/debug
    meta["unbalanced_display_math"] = count_unbalanced_display_math(s)
    meta["brace_badness"] = _basic_brace_balance_score(s[:8000])  # limit for speed
    meta["len"] = len(s)
    return s, meta

def score_markdown(md: str, ocr_meta: Optional[Dict[str, Any]] = None, post_meta: Optional[Dict[str, Any]] = None) -> float:
    """
    Cheap "validator" score to choose the best candidate among base/two-col/hi-DPI.
    Higher is better.
    """
    s = (md or "").strip()
    if not s:
        return -1e9

    score = float(len(s))

    # Strong penalties
    if looks_bad(s):
        score -= 5000.0

    if looks_like_layout(s):
        score -= 2000.0

    # Prefer balanced math fences
    ub = 0
    if post_meta and isinstance(post_meta.get("unbalanced_display_math"), int):
        ub = int(post_meta["unbalanced_display_math"])
    else:
        ub = count_unbalanced_display_math(s)
    score -= 1500.0 * ub

    # Penalize obvious brace mismatch a bit
    bb = 0
    if post_meta and isinstance(post_meta.get("brace_badness"), int):
        bb = int(post_meta["brace_badness"])
    else:
        bb = _basic_brace_balance_score(s[:8000])
    score -= 40.0 * bb

    # Penalize truncation
    if ocr_meta and isinstance(ocr_meta.get("full_meta"), dict):
        fm = ocr_meta["full_meta"]
        if fm.get("truncated") is True:
            score -= 2500.0
        if fm.get("finish_reason") == "length":
            score -= 2000.0

    return score


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type(Exception),
)
def ocr_image_to_markdown(
    client: OpenAI,
    model: str,
    img: Image.Image,
    max_tokens: Optional[int] = None,
    trunc_retry_tokens: int = 4096,
    enable_truncation_retry: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    url = pil_image_to_data_url(img)

    def _call(prompt: str, mt: Optional[int]) -> Tuple[str, Optional[str]]:
        kwargs = dict(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ],
            temperature=0.0,
            top_p=1.0,
        )
        if mt is not None:
            kwargs["max_tokens"] = int(mt)
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = strip_code_fences(choice.message.content or "").strip()
        finish = getattr(choice, "finish_reason", None)
        return text, finish

    meta: Dict[str, Any] = {
        "used_fallback": False,
        "layout": False,
        "boilerplate_stripped": False,
        "finish_reason": None,
        "truncated": False,
        "truncation_retry_used": False,
    }

    md, finish = _call(OCR_MD_PROMPT, max_tokens)
    meta["finish_reason"] = finish
    if finish == "length":
        meta["truncated"] = True

    if enable_truncation_retry and finish == "length":
        retry_mt = int(trunc_retry_tokens if max_tokens is None else min(8192, int(max_tokens * 3 // 2) + 256))
        try:
            md2, finish2 = _call(OCR_MD_PROMPT, retry_mt)
            meta["truncation_retry_used"] = True
            meta["truncation_retry_max_tokens"] = retry_mt
            meta["finish_reason_retry"] = finish2
            if md2 and len(md2) > len(md):
                md, finish = md2, finish2
                meta["finish_reason"] = finish2
                meta["truncated"] = (finish2 == "length")
        except Exception as e:
            meta["truncation_retry_error"] = str(e)

    if looks_like_layout(md):
        meta["layout"] = True
        md2 = strip_layout_boxes(md)
        if len(md2) > 50:
            md = md2

    before = md
    md = strip_boilerplate(md)
    if md != before:
        meta["boilerplate_stripped"] = True

    bad_now = True if meta.get("truncated") else looks_bad(md)
    if bad_now:
        meta["used_fallback"] = True
        md, finish_fb = _call(OCR_MD_PROMPT_FALLBACK, max_tokens)
        meta["finish_reason_fallback"] = finish_fb
        if finish_fb == "length":
            meta["truncated_fallback"] = True

        if looks_like_layout(md):
            meta["layout"] = True
            md2 = strip_layout_boxes(md)
            if len(md2) > 50:
                md = md2

        before = md
        md = strip_boilerplate(md)
        if md != before:
            meta["boilerplate_stripped"] = True

    return md.strip(), meta


def ocr_two_columns(
    client: OpenAI,
    model: str,
    img: Image.Image,
    overlap: int = 30,
    max_tokens: Optional[int] = None,
    trunc_retry_tokens: int = 4096,
    enable_truncation_retry: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    left, right = split_two_columns(img, overlap=overlap)
    md_l, meta_l = ocr_image_to_markdown(
        client, model, left,
        max_tokens=max_tokens,
        trunc_retry_tokens=trunc_retry_tokens,
        enable_truncation_retry=enable_truncation_retry,
    )
    md_r, meta_r = ocr_image_to_markdown(
        client, model, right,
        max_tokens=max_tokens,
        trunc_retry_tokens=trunc_retry_tokens,
        enable_truncation_retry=enable_truncation_retry,
    )
    md = (md_l.strip() + "\n\n" + md_r.strip()).strip()
    meta = {
        "two_col": True,
        "overlap": overlap,
        "left": meta_l,
        "right": meta_r,
        "md_len": len(md),
    }
    return md, meta


# ---------------- main ----------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=str, help="Input PDF")
    ap.add_argument("out_md", type=str, help="Output Markdown file")

    ap.add_argument("--max-tokens", type=int, default=None, help="Override settings.OCR_MAX_TOKENS")
    ap.add_argument("--workers", type=int, default=None, help="Override settings.OCR_WORKERS")
    ap.add_argument("--debug", action="store_true", help="Override settings.OCR_DEBUG=True for this run")
    ap.add_argument("--no-debug", action="store_true", help="Override settings.OCR_DEBUG=False for this run")

    args = ap.parse_args()

    cfg = load_config()
    api_key = require_str(cfg, "api_key")
    base_url = get_cfg(cfg, "base_url", "https://aihubmix.com/v1")
    model = require_str(cfg, "model")

    # default DPI (higher, if config.json doesn't specify)
    settings = load_settings()
    dpi = int(get_setting(settings, "OCR_DPI", 450))

    pdf_path = Path(args.pdf).expanduser().resolve()
    out_md = Path(args.out_md).expanduser().resolve()
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(2)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    workers = int(get_setting(settings, "OCR_WORKERS", 4))
    timeout_s = int(get_setting(settings, "OCR_TIMEOUT", 60))
    max_tokens_setting = get_setting(settings, "OCR_MAX_TOKENS", None)

    debug_enabled = bool(get_setting(settings, "OCR_DEBUG", False))

    # padding (keep small; helps edge clipping)
    pad_top = int(get_setting(settings, "OCR_PAD_TOP", 40))
    pad_other = int(get_setting(settings, "OCR_PAD_OTHER", 8))

    # hi-dpi fallback
    enable_hi_dpi = bool(get_setting(settings, "OCR_ENABLE_HI_DPI", True))
    hi_dpi = int(get_setting(settings, "OCR_HI_DPI", max(dpi, 750)))

    # truncation detection/retry
    enable_truncation_retry = bool(get_setting(settings, "OCR_ENABLE_TRUNCATION_RETRY", True))
    trunc_retry_tokens = int(get_setting(settings, "OCR_TRUNCATION_RETRY_TOKENS", 4096))

    # two-column fallback (tri-state): "off" | "auto" | "on"
    two_col_mode = str(get_setting(settings, "OCR_TWO_COLUMN_MODE", "auto")).lower().strip()
    two_col_overlap = int(get_setting(settings, "OCR_TWO_COLUMN_OVERLAP", 30))

    # postprocess toggle (default ON)
    enable_postprocess = bool(get_setting(settings, "OCR_ENABLE_POSTPROCESS", True))

    # ---- CLI overrides ----
    if args.workers is not None:
        workers = args.workers
    if workers <= 0:
        workers = 4

    if args.max_tokens is not None:
        ocr_max_tokens = args.max_tokens
    else:
        ocr_max_tokens = max_tokens_setting

    if args.debug:
        debug_enabled = True
    if args.no_debug:
        debug_enabled = False

    # Thread-local OpenAI client (safer under multi-thread gateways)
    tls = threading.local()

    def get_client() -> OpenAI:
        c = getattr(tls, "client", None)
        if c is None:
            tls.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
            c = tls.client
        return c

    print(f"[1/2] Render PDF -> images via pdf2image @ {dpi} DPI (no cropping)")
    images = render_pdf_to_pil_images(pdf_path, dpi=dpi)

    print(
        f"[2/2] OCR images -> Markdown (model={model}, workers={workers}, "
        f"pad_top={pad_top}, pad_other={pad_other}, "
        f"two_col={two_col_mode}(overlap={two_col_overlap}), "
        f"hi_dpi={'on' if enable_hi_dpi else 'off'}({hi_dpi}), "
        f"postprocess={'on' if enable_postprocess else 'off'})"
    )

    debug_dir = out_md.parent / f"{out_md.stem}_debug"
    pages_dir = debug_dir / "pages"
    per_page_md_dir = debug_dir / "page_md"
    cols_dir = debug_dir / "columns"

    results: Dict[int, str] = {}
    quality: Dict[int, Dict[str, Any]] = {}

    def prepare_ocr_image(raw_img: Image.Image) -> Image.Image:
        # NO CROPPING: only padding
        return pad_image(raw_img, pad_top=pad_top, pad_other=pad_other)

    def ocr_once(page_i: int, raw_img: Image.Image, dpi_used: int) -> Tuple[str, Dict[str, Any]]:
        ocr_img = prepare_ocr_image(raw_img)

        if debug_enabled:
            save_debug_images(pages_dir, page_i, raw_img, ocr_img, tag=f"dpi{dpi_used}")

        client = get_client()
        md, meta = ocr_image_to_markdown(
            client,
            model,
            ocr_img,
            max_tokens=ocr_max_tokens,
            trunc_retry_tokens=trunc_retry_tokens,
            enable_truncation_retry=enable_truncation_retry,
        )
        out_meta: Dict[str, Any] = {
            "page": page_i,
            "dpi": dpi_used,
            "pad_top": pad_top,
            "pad_other": pad_other,
            "full_meta": meta,
            "md_len_raw": len(md),
        }
        if enable_postprocess:
            md2, pm = postprocess_page_markdown(md)
            out_meta["postprocess"] = pm
            md = md2
        out_meta["md_len"] = len(md)
        return md.strip(), out_meta

    def try_two_column_fallback(page_i: int, raw_img: Image.Image, dpi_used: int) -> Tuple[str, Dict[str, Any]]:
        ocr_img = prepare_ocr_image(raw_img)
        left, right = split_two_columns(ocr_img, overlap=two_col_overlap)
        if debug_enabled:
            save_debug_column_images(cols_dir, page_i, left, right, tag=f"dpi{dpi_used}")

        client = get_client()
        md2, meta2 = ocr_two_columns(
            client,
            model,
            ocr_img,
            overlap=two_col_overlap,
            max_tokens=ocr_max_tokens,
            trunc_retry_tokens=trunc_retry_tokens,
            enable_truncation_retry=enable_truncation_retry,
        )
        info: Dict[str, Any] = {
            "used_two_column_split": True,
            "dpi": dpi_used,
            "two_column_meta": meta2,
            "md_len_raw": len(md2),
        }
        if enable_postprocess:
            md3, pm = postprocess_page_markdown(md2)
            info["postprocess"] = pm
            md2 = md3
        info["md_len"] = len(md2)
        return md2.strip(), info

    def choose_best_candidate(cands: List[Tuple[str, Dict[str, Any]]]) -> Tuple[str, Dict[str, Any]]:
        best_md = ""
        best_meta: Dict[str, Any] = {}
        best_score = -1e18

        for md, meta in cands:
            pm = meta.get("postprocess") if isinstance(meta.get("postprocess"), dict) else None
            sc = score_markdown(md, ocr_meta=meta, post_meta=pm)
            meta["_score"] = sc
            if sc > best_score:
                best_score = sc
                best_md = md
                best_meta = meta
        best_meta["_best_score"] = best_score
        best_meta["_num_candidates"] = len(cands)
        return best_md, best_meta

    def should_trigger_hi_dpi(md: str, meta: Dict[str, Any]) -> bool:
        # More general triggers than looks_bad():
        if looks_bad(md):
            return True
        pm = meta.get("postprocess")
        if isinstance(pm, dict) and int(pm.get("unbalanced_display_math", 0)) != 0:
            return True
        if isinstance(pm, dict) and int(pm.get("brace_badness", 0)) >= 8:
            return True
        fm = meta.get("full_meta")
        if isinstance(fm, dict) and fm.get("truncated") is True:
            return True
        return False

    def process_page(page_i: int, im: Image.Image) -> Tuple[int, str, Dict[str, Any]]:
        candidates: List[Tuple[str, Dict[str, Any]]] = []

        # Base
        md_base, meta_base = ocr_once(page_i, im, dpi_used=dpi)
        meta_base["variant"] = "base"
        candidates.append((md_base, meta_base))

        # Optional: two-column fallback (candidate)
        if two_col_mode != "off":
            ocr_img_tmp = prepare_ocr_image(im)
            if two_col_mode == "on":
                trigger_two_col = True
            else:
                img_two_col = is_likely_two_column_image(ocr_img_tmp)
                trigger_two_col = img_two_col and (looks_like_interleaved_columns(md_base) or looks_bad(md_base))

            if trigger_two_col:
                try:
                    md_two, meta_two = try_two_column_fallback(page_i, im, dpi_used=dpi)
                    meta_two["variant"] = "two_col"
                    meta_two["two_col_mode"] = two_col_mode
                    candidates.append((md_two, meta_two))
                except Exception as e:
                    meta_base["two_column_error"] = str(e)
                    meta_base["two_col_mode"] = two_col_mode

        # Optional: hi-DPI candidate if triggered by general signals
        if enable_hi_dpi and should_trigger_hi_dpi(md_base, meta_base):
            try:
                rer = render_single_page(pdf_path, dpi=hi_dpi, page_1based=page_i)
                md_hi, meta_hi = ocr_once(page_i, rer, dpi_used=hi_dpi)
                meta_hi["variant"] = "hi_dpi"
                candidates.append((md_hi, meta_hi))

                # hi-DPI + two-col candidate (if enabled)
                if two_col_mode != "off":
                    ocr_img_hi = prepare_ocr_image(rer)
                    if two_col_mode == "on":
                        trigger_two_col_hi = True
                    else:
                        img_two_col_hi = is_likely_two_column_image(ocr_img_hi)
                        trigger_two_col_hi = img_two_col_hi and (looks_like_interleaved_columns(md_hi) or looks_bad(md_hi))

                    if trigger_two_col_hi:
                        try:
                            md_hi_two, meta_hi_two = try_two_column_fallback(page_i, rer, dpi_used=hi_dpi)
                            meta_hi_two["variant"] = "hi_dpi_two_col"
                            meta_hi_two["two_col_mode"] = two_col_mode
                            candidates.append((md_hi_two, meta_hi_two))
                        except Exception as e:
                            meta_hi["two_column_error_hi_dpi"] = str(e)
            except Exception as e:
                meta_base["hi_dpi_error"] = str(e)

        # Choose best by score (general)
        md_best, meta_best = choose_best_candidate(candidates)

        # debug write
        if debug_enabled:
            per_page_md_dir.mkdir(parents=True, exist_ok=True)
            (per_page_md_dir / f"page_{page_i:04d}.md").write_text(md_best, encoding="utf-8")
            # also dump candidates scores
            cand_dump = {
                "page": page_i,
                "candidates": [
                    {
                        "variant": m.get("variant"),
                        "dpi": m.get("dpi"),
                        "score": m.get("_score"),
                        "len": len(t),
                        "postprocess": m.get("postprocess"),
                        "full_meta": m.get("full_meta"),
                    }
                    for (t, m) in candidates
                ],
                "chosen": {
                    "variant": meta_best.get("variant"),
                    "dpi": meta_best.get("dpi"),
                    "score": meta_best.get("_best_score"),
                }
            }
            (per_page_md_dir / f"page_{page_i:04d}_candidates.json").write_text(
                json.dumps(cand_dump, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        pm_best = meta_best.get("postprocess") if isinstance(meta_best.get("postprocess"), dict) else {}
        bad = looks_bad(md_best)
        layout = False
        fm = meta_best.get("full_meta")
        if isinstance(fm, dict):
            layout = bool(fm.get("layout"))
        print(
            f"[OCR] page={page_i} bad={bad} layout={layout} len={len(md_best)} "
            f"dpi={meta_best.get('dpi')} variant={meta_best.get('variant')} "
            f"eqnum_moved={pm_best.get('eqnum_moved', 0)} "
            f"unbal$$={pm_best.get('unbalanced_display_math', 0)} "
            f"score={meta_best.get('_best_score')}"
        )
        return page_i, md_best, meta_best

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_page, i, im): i for i, im in enumerate(images, start=1)}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="OCR pages"):
            page_i, md, meta = fut.result()
            results[page_i] = md
            quality[page_i] = meta

    chunks: List[str] = []
    for i in range(1, len(images) + 1):
        md = results.get(i, "")
        chunks.append(f"<!-- PAGE {i} -->\n{md}\n")

    out_md.write_text("\n".join(chunks), encoding="utf-8")

    if debug_enabled:
        debug_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "pdf": str(pdf_path),
            "out_md": str(out_md),
            "dpi": dpi,
            "workers": workers,
            "timeout_s": timeout_s,
            "model": model,
            "pad_top": pad_top,
            "pad_other": pad_other,
            "enable_hi_dpi": enable_hi_dpi,
            "hi_dpi": hi_dpi if enable_hi_dpi else None,
            "two_col_mode": two_col_mode,
            "two_col_overlap": two_col_overlap,
            "enable_truncation_retry": enable_truncation_retry,
            "trunc_retry_tokens": trunc_retry_tokens,
            "enable_postprocess": enable_postprocess,
            "pages": [quality[k] for k in sorted(quality.keys())],
        }
        (debug_dir / "quality_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"DONE: {out_md}")
    if debug_enabled:
        print(f"DEBUG: {debug_dir}")


if __name__ == "__main__":
    main()
