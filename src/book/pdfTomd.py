#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Book-oriented PDF to Markdown OCR."""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageOps
import pdf2image
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI


OCR_MD_PROMPT = """
You are doing OCR for pages from the book "Convex Optimization Solutions Manual".
Return only Markdown for the main mathematical content on the page.

The page often contains:
- repeated running headers such as "Exercises", chapter/page headers such as "2 Convex sets" or "Chapter 3",
- numbered exercises such as "2.1", "2.29",
- markers such as "Solution.", "Proof.", "Hint.", and parts "(a)", "(b)", "(c)",
- displayed equations, matrices, bullets, short derivations, and sometimes Matlab code,
- figures that may include PSfrag artifacts or isolated labels.

Rules:
0) HARD CONSTRAINT FOR EXERCISE HEADINGS:
        - The ONLY allowed exercise heading format is: **Exercise x.y** <rest of line>
        - Any Markdown heading form for exercises is FORBIDDEN, including "#", "##", "###".
        - If recognized text looks like "### 2.1 ..." or "### Exercise 2.1 ...", rewrite it to
            "**Exercise 2.1** ..." before output.

1) Transcribe the main body faithfully, but OMIT obvious running headers/footers/page boilerplate when they are visually separate from the body content.
    Examples of boilerplate to omit: repeated "Exercises", repeated chapter/page headers at the top of the page, isolated page-top duplicates.

2) For exercise lines, normalize the visible number to this format:
      **Exercise x.y** <rest of the line>
    Examples:
      - "**2.1** Let C be convex ..." -> "**Exercise 2.1** Let C be convex ..."
      - "**2.10 Solution set ...**" -> "**Exercise 2.10** Solution set ..."
        Exercise headings MUST remain in the bold inline form above, never as Markdown headings.
    Do NOT invent exercise numbers; only normalize numbers that are clearly visible.

3) Keep real body headings such as section/subsection titles when they belong to the content
    (for example: "Definition of convexity", "Examples", "Operations that preserve convexity").

4) Keep "Solution.", "Proof.", "Hint.", and sub-question markers "(a)", "(b)", "(c)", ...
    exactly when visible. Do not invent missing parts.
    Prefer a standalone line exactly "Solution." for solution headings when possible.
    If a line contains both a sub-question marker and "Solution.", keep them as separate lines
    so downstream parsing can distinguish statement vs. solution.
    For cross-page continuations (e.g., (b)/(c)/(d) on next page), preserve these markers faithfully.
    Special handling A (interleaved multi-part exercises like 2.17):
    when you see patterns such as "(b) ...", "Solution.", "(c) ...", "Solution." across page boundaries,
    preserve each "(part) statement" and each "Solution." boundary explicitly.
    Do not merge "(b)/(c)/(d)" statements into the previous part's proof text.
    Special handling B (solution continues to next page):
    if a page starts with "(b)" (or another part marker) after a previous-page "Solution.",
    treat it as a continuation of the same exercise solution block, not a new exercise.
    Ignore running headers in between (e.g., "Exercises", "2 Convex sets").

5) Preserve mathematics faithfully using standard Markdown math:
    - inline math: $...$
    - display math: $$...$$
    Preserve matrices, norms, subscripts, superscripts, transpose notation, cone/order symbols,
    and inequalities as faithfully as possible.

6) Preserve readable line structure for multi-line displayed derivations and aligned calculations.
    Do not collapse a derivation into one long sentence.

7) If the page contains program code (for example Matlab), transcribe it verbatim as plain text.
    Do not explain it, summarize it, or convert it to pseudocode.

8) Do NOT output layout metadata, coordinates, bounding boxes, or OCR control text.

9) Figures:
    - If a figure only contributes noisy OCR artifacts such as "PSfrag replacements" or scattered labels,
      do NOT dump the raw artifact text.
    - If the nearby text explicitly refers to a figure and the figure contains a few clearly readable labels
      that are genuinely helpful, you may replace the figure by a SHORT placeholder such as:
         [Figure]
         [Figure: cone K with rays α, β]
    - Do not hallucinate detailed figure descriptions.

10) Do NOT hallucinate, complete missing text, merge repeated lines, or rewrite the mathematics.
     If a short local fragment is unreadable, use [illegible].

11) Output only the page content. No comments, no explanations, no surrounding prose,
     and do not wrap the whole response in a code fence.
""".strip()

OCR_MD_PROMPT_FALLBACK = """
Conservative OCR mode for a Convex Optimization Solutions Manual page.

Return only Markdown for clearly readable MAIN BODY content.

Rules:
0) HARD CONSTRAINT FOR EXERCISE HEADINGS:
    - NEVER output exercise titles as Markdown headings (e.g., "### 2.1", "### Exercise 2.1").
    - ALWAYS rewrite them to: **Exercise x.y** <rest of line>.

1) Omit obvious running headers/footers/page boilerplate such as repeated "Exercises",
    repeated chapter/page headers, and top-of-page duplicates.

2) For exercise lines, normalize the visible number to this format:
      **Exercise x.y** <rest of the line>
    Any exercise-like heading with "# / ## / ###" must be converted to this inline bold format.
    Do NOT invent exercise numbers; only normalize numbers that are clearly visible.

3) Keep "Solution.", "Proof.", "Hint.", and markers "(a)", "(b)", "(c)" when clearly visible.
   Prefer a standalone line exactly "Solution." for solution headings when possible.
   If a line contains both a sub-question marker and "Solution.", split them into separate lines.
   For cross-page continuations (e.g., (b)/(c)/(d) on next page), preserve markers as-is.
   Special handling A (2.17-like interleaving): keep each "(part) statement" and each "Solution." as
   explicit boundaries, even when they alternate across pages.
   Special handling B (page-break continuation): if "(b)" appears at top of page after prior "Solution.",
   keep it as same exercise continuation; do not create a new exercise heading.
   Ignore running headers ("Exercises", chapter title lines) between these boundaries.

4) Preserve visible math faithfully with $...$ and $$...$$.

5) Do not output raw "PSfrag replacements", layout coordinates, or OCR meta text.

6) If a figure is mostly non-text or noisy, omit it or use a minimal placeholder [Figure].

7) Do not infer missing text, do not repeat lines, do not paraphrase, and do not add explanations.
    If a short fragment is unreadable, use [illegible].

8) Output only content, with no surrounding prose and no full-response code fence.
""".strip()


# -------- config helpers --------

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


def load_config() -> Dict[str, Any]:
    cfg_path = find_config_json()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{cfg_path} must contain a JSON object.")
    return data


def require_str(cfg: Dict[str, Any], key: str) -> str:
    v = cfg.get(key)
    if not isinstance(v, str) or not v.strip():
        raise KeyError(f"Missing/invalid '{key}' in config.json (must be non-empty string)")
    return v.strip()


def get_cfg(
    cfg: Dict[str, Any],
    key: str,
    default: Any,
    *,
    expected_type: Optional[type] = None,
    nonempty: bool = False,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_none: bool = False,
) -> Any:
    """
    Typed config getter with optional range checks.
    Keeps config.json structure unchanged, but enforces types/ranges when requested.
    """
    v = cfg.get(key, default)
    if v is None and allow_none:
        return None

    if expected_type is not None and not isinstance(v, expected_type):
        raise TypeError(f"Invalid '{key}' in config.json: expected {expected_type.__name__}, got {type(v).__name__}")

    if isinstance(v, str) and nonempty and not v.strip():
        raise ValueError(f"Invalid '{key}' in config.json: must be non-empty string")

    if isinstance(v, (int, float)):
        if min_value is not None and float(v) < float(min_value):
            raise ValueError(f"Invalid '{key}' in config.json: {v} < min {min_value}")
        if max_value is not None and float(v) > float(max_value):
            raise ValueError(f"Invalid '{key}' in config.json: {v} > max {max_value}")

    return v


# -------- settings helpers (module-local JSON) --------

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


# -------- PDF render (stream per page) --------

def get_pdf_page_count(pdf_path: Path) -> int:
    info = pdf2image.pdfinfo_from_path(str(pdf_path))
    pages = info.get("Pages")
    if not isinstance(pages, int) or pages <= 0:
        raise RuntimeError(f"Failed to read PDF page count: {pdf_path}")
    return pages


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


# -------- image utilities --------

def pad_image(img: Image.Image, pad_top: float = 0.0, pad_other: float = 0.0, *, max_px: int = 120) -> Image.Image:
    """
    Add white padding (helps OCR not miss edge-touching glyphs).
    Padding is now proportional to image size by default.

    pad_top / pad_other:
      - if 0 < value <= 1: treated as ratio of image height (top) / min(w,h) (other)
      - if value > 1: treated as pixels (compat)
    """
    w, h = img.size

    def _to_px(v: float, *, base: int) -> int:
        if v <= 0:
            return 0
        if 0 < v <= 1:
            return int(round(base * v))
        return int(round(v))

    top_px = _to_px(float(pad_top), base=h)
    other_base = min(w, h)
    other_px = _to_px(float(pad_other), base=other_base)

    if max_px > 0:
        top_px = min(top_px, int(max_px))
        other_px = min(other_px, int(max_px))

    if top_px <= 0 and other_px <= 0:
        return img

    border = (other_px, top_px, other_px, other_px)  # left, top, right, bottom
    return ImageOps.expand(img, border=border, fill="white")


def save_debug_images(
    debug_dir: Path,
    page_i: int,
    raw_img: Image.Image,
    ocr_img: Image.Image,
    tag: str,
    *,
    jpeg_quality: int = 90,
) -> None:
    """
    Save images for debugging (JPEG only):
      - raw render
      - final OCR input (after padding)
    """
    debug_dir.mkdir(parents=True, exist_ok=True)
    q = int(max(1, min(95, jpeg_quality)))
    try:
        raw_img.save(debug_dir / f"page_{page_i:04d}_{tag}_raw.jpg", "JPEG", quality=q, optimize=True, progressive=True)
        ocr_img.save(debug_dir / f"page_{page_i:04d}_{tag}_ocr.jpg", "JPEG", quality=q, optimize=True, progressive=True)
    except Exception:
        pass


# -------- OCR helpers --------

def _default_upload_max_side_for_dpi(dpi: int) -> int:
    """
    Heuristic: larger DPI -> allow larger upload max_side.
    Clamped to keep payload manageable.
    """
    # 350 dpi -> ~2100; 450 -> ~2700; 600 -> ~3600 (clamp)
    m = int(round(dpi * 6))
    return max(1800, min(3600, m))


def _downscale_if_needed(img: Image.Image, max_side: Optional[int]) -> Image.Image:
    """
    Downscale image so that max(width, height) <= max_side (keeps aspect ratio).
    Uses high-quality resampling. If max_side is None or <=0, no-op.
    """
    if not max_side or max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)


def pil_image_to_data_url(
    img: Image.Image,
    *,
    fmt: str = "JPEG",                 # "PNG" | "JPEG"
    jpeg_quality: int = 85,            # 1..95
    max_side: Optional[int] = 2000,    # int or None
    grayscale: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Encode image to data URL for API upload.
    Returns (data_url, info) where info includes actual upload size/bytes.
    """
    orig_w, orig_h = img.size

    img2 = _downscale_if_needed(img, max_side=max_side)
    up_w, up_h = img2.size

    if grayscale:
        img2 = img2.convert("L")
    else:
        if img2.mode != "RGB":
            img2 = img2.convert("RGB")

    buf = io.BytesIO()
    fmt_u = (fmt or "JPEG").upper()

    if fmt_u == "PNG":
        img2.save(buf, format="PNG", optimize=True)
        mime = "image/png"
        actual_fmt = "PNG"
        q = None
    else:
        q = int(max(1, min(95, int(jpeg_quality))))
        img2.save(
            buf,
            format="JPEG",
            quality=q,
            optimize=True,
            progressive=True,
        )
        mime = "image/jpeg"
        actual_fmt = "JPEG"

    raw_bytes = buf.getvalue()
    b64 = base64.b64encode(raw_bytes).decode("ascii")

    info: Dict[str, Any] = {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "upload_w": up_w,
        "upload_h": up_h,
        "upload_mode": "L" if grayscale else "RGB",
        "upload_fmt": actual_fmt,
        "jpeg_quality": q,
        "max_side": max_side,
        "grayscale": grayscale,
        "upload_bytes": len(raw_bytes),
        "upload_b64_chars": len(b64),
    }

    return f"data:{mime};base64,{b64}", info


def strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^\s*```(?:md|markdown)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def looks_like_layout(md: str) -> bool:
    s = (md or "").strip()
    if not s:
        return True
    head = s[:1000]
    return bool(re.search(
        r"\b(text|equation|interline_equation|sub_title|title|figure|table)\s*\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]",
        head
    ))


def strip_layout_boxes(md: str) -> str:
    lines = (md or "").splitlines()
    out = []
    for ln in lines:
        ln2 = re.sub(r"^\s*\w+\s*\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]\s*", "", ln)
        if ln2.strip():
            out.append(ln2.rstrip())
    return "\n".join(out).strip()


def compile_boilerplate_patterns(patterns: Any) -> List[re.Pattern]:
    """
    patterns: list[str] regex strings.
    """
    out: List[re.Pattern] = []
    if not patterns:
        return out
    if not isinstance(patterns, list):
        return out
    for p in patterns:
        if not isinstance(p, str) or not p.strip():
            continue
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except re.error:
            continue
    return out


def strip_boilerplate(md: str, patterns: List[re.Pattern]) -> Tuple[str, int]:
    """
    Remove leading boilerplate lines that match any configured pattern.
    Returns (cleaned_md, removed_count).
    """
    lines = (md or "").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)

    removed = 0
    # remove up to 6 leading boilerplate lines (more flexible than fixed 3)
    while lines and removed < 6:
        head = lines[0]
        if any(rx.match(head) for rx in patterns):
            lines.pop(0)
            removed += 1
            while lines and not lines[0].strip():
                lines.pop(0)
        else:
            break

    return "\n".join(lines).strip(), removed


MIDPAGE_NOISE_LINE_PATTERNS = [
    re.compile(r"^\s*#*\s*Exercises\s*$", re.IGNORECASE),
    re.compile(r"^\s*#*\s*Chapter\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*#*\s*\d+\s+Convex sets\s*$", re.IGNORECASE),
    re.compile(r"^\s*#*\s*\d+\s+Convex functions\s*$", re.IGNORECASE),
    re.compile(r"^\s*#*\s*\d+\s+Duality\s*$", re.IGNORECASE),
    re.compile(r"^\s*#*\s*Generalized inequalities\s*$", re.IGNORECASE),
]


def strip_midpage_running_headers(md: str) -> Tuple[str, int]:
    """
    Remove very obvious running-header noise that appears in the middle of content.
    Be conservative: only drop isolated lines or glued boundary artifacts.
    """
    if not md:
        return "", 0

    lines = md.splitlines()
    out: List[str] = []
    removed = 0

    for ln in lines:
        s = (ln or "").strip()

        # 1) Drop isolated header-like noise lines anywhere in the page
        if s and any(rx.match(s) for rx in MIDPAGE_NOISE_LINE_PATTERNS):
            removed += 1
            continue

        out.append(ln)

    s = "\n".join(out)

    # 2) Fix glued artifacts like:
    #    "... p*.Exercises 2.35 ..."
    #    "... 0.Exercises\n2.35 ..."
    s2 = re.sub(
        r"(?<=[A-Za-z0-9\.\)\]])\s*Exercises\s+(?=(?:\d+\.\d+\b|[A-Z]\w))",
        " ",
        s,
        flags=re.IGNORECASE,
    )

    # 3) Fix glued chapter/page headers inside prose boundaries
    s2 = re.sub(
        r"(?<=[A-Za-z0-9\.\)\]])\s*\d+\s+Convex sets\s+(?=(?:\d+\.\d+\b|[A-Z]\w))",
        " ",
        s2,
        flags=re.IGNORECASE,
    )
    s2 = re.sub(
        r"(?<=[A-Za-z0-9\.\)\]])\s*\d+\s+Duality\s+(?=(?:\d+\.\d+\b|[A-Z]\w))",
        " ",
        s2,
        flags=re.IGNORECASE,
    )

    if s2 != s:
        removed += 1

    return s2.strip(), removed


def has_prompt_leakage(s: str) -> bool:
    text = (s or "").lower()
    # Common instruction leakage fragments from OCR prompts / conversion prompts.
    leak_markers = [
        "the output must be latex",
        "output markdown only",
        "output only the content",
        "strict block rules",
        "placeholder tokens",
        "do not move placeholders",
        "extract all text and math",
        "you are doing ocr",
    ]
    return any(m in text for m in leak_markers)


def has_runaway_number_list(s: str) -> bool:
    # Catch pathological "1. 2. 3. ... 500." loops.
    return bool(re.search(r"(?:\b\d+\.\s*){120,}", s or ""))


def has_heavy_line_repetition(s: str) -> bool:
    lines = [(ln or "").strip() for ln in (s or "").splitlines()]
    lines = [ln for ln in lines if ln]
    if len(lines) < 12:
        return False

    counts: Dict[str, int] = {}
    for ln in lines:
        key = re.sub(r"\s+", " ", ln).lower()
        if len(key) < 24:
            continue
        counts[key] = counts.get(key, 0) + 1

    if not counts:
        return False
    max_rep = max(counts.values())
    # "same long line repeated many times" is almost always OCR degeneration.
    if max_rep >= 6:
        return True
    # Or one line dominates too much.
    if max_rep / max(1, len(lines)) >= 0.25:
        return True
    return False


def _bad_reason_penalty(reason: Optional[str]) -> int:
    penalties = {
        "empty": 5000,
        "prompt_echo": 4500,
        "runaway_number_list": 4200,
        "line_repetition": 3800,
        "sub-loop": 3000,
        "excessive_quad": 2000,
        "excessive_newlines": 1500,
        "char_repetition": 1500,
        "unbalanced_braces": 900,
    }
    return penalties.get(reason or "", 1000 if reason else 0)


def score_candidate(md: str, flags: Dict[str, Any]) -> int:
    """
    Higher is better.
    Prefer non-bad outputs, then prefer richer but not absurdly long text.
    """
    text = (md or "").strip()
    score = 0

    if flags.get("bad"):
        score -= _bad_reason_penalty(flags.get("bad_reason"))
    else:
        score += 2000

    n = len(text)
    if n < 20:
        score -= 1000
    elif n < 80:
        score -= 300
    elif n <= 14000:
        score += min(1800, n // 6)
    else:
        # Too long on a single page usually means degeneration.
        score -= min(2200, (n - 14000) // 10)

    if flags.get("layout"):
        score -= 80

    # Additional penalties for common corruption patterns.
    if has_prompt_leakage(text):
        score -= 3000
    if has_runaway_number_list(text):
        score -= 3000
    if has_heavy_line_repetition(text):
        score -= 2200

    return int(score)


def pick_better_candidate(
    a_md: str,
    a_flags: Dict[str, Any],
    b_md: str,
    b_flags: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], bool]:
    """
    Returns (best_md, best_flags, picked_b).
    """
    sa = score_candidate(a_md, a_flags)
    sb = score_candidate(b_md, b_flags)
    if sb > sa:
        return b_md, b_flags, True
    return a_md, a_flags, False


def postprocess_and_assess(md: str, boilerplate_patterns: List[re.Pattern]) -> Tuple[str, Dict[str, Any]]:
    """
    Merge: strip_code_fences + layout stripping + configurable boilerplate stripping + bad detection.
    Returns (clean_md, meta_flags).
    """
    meta: Dict[str, Any] = {
        "layout": False,
        "boilerplate_removed_lines": 0,
        "midpage_noise_removed": 0,
        "bad": False,
        "bad_reason": None,
    }

    s = strip_code_fences(md or "")

    if looks_like_layout(s):
        meta["layout"] = True
        s2 = strip_layout_boxes(s)
        if len(s2) > 50:
            s = s2

    s, removed = strip_boilerplate(s, boilerplate_patterns)
    meta["boilerplate_removed_lines"] = removed

    s, removed_mid = strip_midpage_running_headers(s)
    meta["midpage_noise_removed"] = removed_mid

    s_stripped = (s or "").strip()

    # --- 1. Basic Check: Empty ---
    if not s_stripped:
        meta["bad"] = True
        meta["bad_reason"] = "empty"
        return "", meta

    head = s_stripped[:800].lower()
    
    # --- 2. Basic Check: Loop/Hallucination ---
    if "sub-sub-sub" in head:
        meta["bad"] = True
        meta["bad_reason"] = "sub-loop"
        return s_stripped, meta

    # --- 3. Prompt leakage / echo ---
    echo_phrases = [
        "ocr to markdown",
        "output markdown only",
        "output only the content",
        "extract all text and math",
    ]
    if (len(s_stripped) < 240 and any(head.startswith(p) for p in echo_phrases)) or has_prompt_leakage(s_stripped):
        meta["bad"] = True
        meta["bad_reason"] = "prompt_echo"
        return s_stripped, meta

    # --- 4. Repetition / loop degeneration ---
    if has_runaway_number_list(s_stripped):
        meta["bad"] = True
        meta["bad_reason"] = "runaway_number_list"
        return s_stripped, meta

    if has_heavy_line_repetition(s_stripped):
        meta["bad"] = True
        meta["bad_reason"] = "line_repetition"
        return s_stripped, meta

    # --- 5. Math degradation checks ---
    if re.search(r"(\\quad\s*){6,}", s_stripped):
        meta["bad"] = True
        meta["bad_reason"] = "excessive_quad"
        return s_stripped, meta
        
    if re.search(r"(\\\\\s*){10,}", s_stripped):
        meta["bad"] = True
        meta["bad_reason"] = "excessive_newlines"
        return s_stripped, meta

    if re.search(r"([a-zA-Z0-9\uff01-\uff5e])\1{19,}", s_stripped):
        meta["bad"] = True
        meta["bad_reason"] = "char_repetition"
        return s_stripped, meta

    open_braces = s_stripped.count('{')
    close_braces = s_stripped.count('}')
    if abs(open_braces - close_braces) > 15:
        meta["bad"] = True
        meta["bad_reason"] = "unbalanced_braces"
        return s_stripped, meta

    return s_stripped, meta


def normalize_exercise_bold_headers(md: str) -> str:
    """
    Normalize exercise heading lines to a single canonical format:
      **Exercise x.y** <optional title/body on same line>

    Handles common OCR variants:
      - **2.1**
      - **2.1** Let C be convex ...
      - **2.10 Solution set ...**
      - 2.10 Solution set ...
      - Exercise 2.10 Solution set ...
            - ### 2.10 Solution set ...
            - ### Exercise 2.10 Solution set ...
    """
    s = md or ""
    lines = s.splitlines()
    out_lines: List[str] = []

    # Already canonical: **Exercise 2.10** ...
    canonical_re = re.compile(r"^(\s*)\*\*Exercise\s+(\d+(?:\.\d+)+)\*\*(.*)$", re.IGNORECASE)
    # Bold-number variants: **2.10** ..., **2.10 title...**, **2.10 title...** ...
    bold_num_re = re.compile(r"^(\s*)\*\*(\d+(?:\.\d+)+)(?:\s+([^*]*?))?\*\*(.*)$")
    # Plain-number variants: 2.10 ..., 2.10: ..., 2.10) ...
    plain_num_re = re.compile(r"^(\s*)(\d+(?:\.\d+)+)\s*[):.]?\s*(.*)$")
    # "Exercise 2.10 ..." (non-bold)
    plain_ex_re = re.compile(r"^(\s*)Exercise\s+(\d+(?:\.\d+)+)\s*(.*)$", re.IGNORECASE)
    # Markdown heading variants to suppress (e.g., ### 2.10 ..., ### Exercise 2.10 ...)
    md_heading_num_re = re.compile(r"^(\s*)#{1,6}\s*(\d+(?:\.\d+)+)\s*[):.]?\s*(.*)$")
    md_heading_ex_re = re.compile(r"^(\s*)#{1,6}\s*Exercise\s+(\d+(?:\.\d+)+)\s*(.*)$", re.IGNORECASE)

    for ln in lines:
        line = ln.rstrip()

        m0 = canonical_re.match(line)
        if m0:
            indent, num, rest = m0.groups()
            rest2 = (rest or "").strip()
            out_lines.append(f"{indent}**Exercise {num}**" + (f" {rest2}" if rest2 else ""))
            continue

        m1 = bold_num_re.match(line)
        if m1:
            indent, num, inner_tail, rest = m1.groups()
            merged_tail = " ".join([x.strip() for x in [inner_tail or "", rest or ""] if x and x.strip()])
            out_lines.append(f"{indent}**Exercise {num}**" + (f" {merged_tail}" if merged_tail else ""))
            continue

        mh1 = md_heading_ex_re.match(line)
        if mh1:
            indent, num, rest = mh1.groups()
            rest2 = (rest or "").strip()
            out_lines.append(f"{indent}**Exercise {num}**" + (f" {rest2}" if rest2 else ""))
            continue

        mh2 = md_heading_num_re.match(line)
        if mh2:
            indent, num, rest = mh2.groups()
            rest2 = (rest or "").strip()
            out_lines.append(f"{indent}**Exercise {num}**" + (f" {rest2}" if rest2 else ""))
            continue

        m2 = plain_ex_re.match(line)
        if m2:
            indent, num, rest = m2.groups()
            rest2 = (rest or "").strip()
            out_lines.append(f"{indent}**Exercise {num}**" + (f" {rest2}" if rest2 else ""))
            continue

        m3 = plain_num_re.match(line)
        if m3:
            indent, num, rest = m3.groups()
            rest2 = (rest or "").strip()
            out_lines.append(f"{indent}**Exercise {num}**" + (f" {rest2}" if rest2 else ""))
            continue

        out_lines.append(line)

    return "\n".join(out_lines)


_SOLUTION_TOKEN_RE = re.compile(r"(?i)\b(?:\*\*|__)?solution(?:\*\*|__)?\s*[:.]")
_SUBPART_TOKEN_RE = re.compile(r"(?i)\(\s*[a-z]\s*\)")
_SOLUTION_LINE_RE = re.compile(r"^\s*(?:\*\*|__)?solution(?:\*\*|__)?\s*[:.]?\s*(.*)$", re.IGNORECASE)


def normalize_subpart_solution_boundaries(md: str) -> str:
    """
    OCR guardrail:
    - if "(b) ... Solution. ..." appears on one line, split before "Solution."
    - if "Solution. ... (b) ..." appears on one line, split before "(b)"
    Keep changes conservative and line-local.
    """
    lines = (md or "").splitlines()
    out: List[str] = []

    for ln in lines:
        s = (ln or "").rstrip()
        if not s.strip():
            out.append(ln)
            continue

        sol = _SOLUTION_TOKEN_RE.search(s)
        sub = _SUBPART_TOKEN_RE.search(s)
        if sol and sub:
            if sub.start() < sol.start():
                left = s[:sol.start()].rstrip()
                right = s[sol.start():].lstrip()
                if left:
                    out.append(left)
                if right:
                    out.append(right)
                continue
            if sol.start() < sub.start():
                left = s[:sub.start()].rstrip()
                right = s[sub.start():].lstrip()
                if left:
                    out.append(left)
                if right:
                    out.append(right)
                continue

        out.append(ln)

    return "\n".join(out).strip()


def normalize_solution_heading_lines(md: str) -> str:
    """
    Canonicalize standalone solution headings to exactly:
      Solution.
    If a heading has same-line tail text, split to two lines:
      Solution.
      <tail>
    """
    lines = (md or "").splitlines()
    out: List[str] = []
    for ln in lines:
        m = _SOLUTION_LINE_RE.match((ln or "").strip())
        if not m:
            out.append(ln)
            continue
        tail = (m.group(1) or "").strip()
        # keep only clear heading-like lines; avoid converting prose mentions.
        if (ln or "").strip().lower().startswith(("solution", "**solution", "__solution")):
            out.append("Solution.")
            if tail:
                out.append(tail)
            continue
        out.append(ln)
    return "\n".join(out).strip()


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
    boilerplate_patterns: List[re.Pattern],
    *,
    max_tokens: Optional[int] = None,
    upload_fmt: str = "JPEG",
    upload_jpeg_quality: int = 85,
    upload_max_side: Optional[int] = 2000,
    upload_grayscale: bool = False,
    api_semaphore: Optional[threading.Semaphore] = None,
    base_url_hint: str = "",
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (markdown, meta) where meta includes flags about fallback/layout/boilerplate and upload info.
    """
    url, upload_info = pil_image_to_data_url(
        img,
        fmt=upload_fmt,                      # FIX: no hardcode
        jpeg_quality=upload_jpeg_quality,    # FIX: no hardcode
        max_side=upload_max_side,            # FIX: no hardcode
        grayscale=upload_grayscale,          # FIX: no hardcode
    )

    base_url_norm = (base_url_hint or "").strip().lower()
    is_codex_gateway = "codex-for.me" in base_url_norm
    _chat_force_stream: Optional[bool] = True if is_codex_gateway else None

    def _is_stream_required_error(err: Exception) -> bool:
        return "stream must be set to true" in str(err or "").lower()

    def _collect_stream_text(stream_obj: Any) -> str:
        parts: List[str] = []
        try:
            for chunk in stream_obj:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for it in content:
                        if isinstance(it, dict):
                            t = it.get("text") or it.get("content") or ""
                        else:
                            t = getattr(it, "text", "") or getattr(it, "content", "") or ""
                        if isinstance(t, str) and t:
                            parts.append(t)
        finally:
            close_fn = getattr(stream_obj, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        return "".join(parts)

    def _collect_responses_text(resp: Any) -> str:
        if isinstance(resp, str):
            return resp.strip()

        if isinstance(resp, dict):
            txt = resp.get("output_text")
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
            out_items = resp.get("output") or []
        else:
            txt = getattr(resp, "output_text", None)
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
            out_items = getattr(resp, "output", None) or []

        parts: List[str] = []
        for item in out_items:
            contents = getattr(item, "content", None)
            if contents is None and isinstance(item, dict):
                contents = item.get("content")
            for c in (contents or []):
                if isinstance(c, dict):
                    t = c.get("text") or c.get("content") or ""
                else:
                    t = getattr(c, "text", "") or getattr(c, "content", "") or ""
                if isinstance(t, str) and t:
                    parts.append(t)
        return "".join(parts).strip()

    def _call_chat(prompt: str) -> str:
        nonlocal _chat_force_stream
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
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if _chat_force_stream is True:
            stream_obj = client.chat.completions.create(stream=True, **kwargs)
            return _collect_stream_text(stream_obj) or ""
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:
            if _is_stream_required_error(e):
                _chat_force_stream = True
                stream_obj = client.chat.completions.create(stream=True, **kwargs)
                return _collect_stream_text(stream_obj) or ""
            raise

    def _call_responses(prompt: str) -> str:
        kwargs = dict(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": url},
                    ],
                }
            ],
        )
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens
        resp = client.responses.create(**kwargs)
        out = _collect_responses_text(resp)
        if not out:
            raise RuntimeError("responses endpoint returned empty output")
        return out

    def _call(prompt: str) -> str:
        # limit API concurrency if semaphore is provided
        if api_semaphore is not None:
            api_semaphore.acquire()
        try:
            # codex-for.me currently returns empty body on /responses; use chat-stream path directly.
            if is_codex_gateway:
                return _call_chat(prompt)
            # Prefer responses API for gateways where chat.completions is unstable.
            try:
                return _call_responses(prompt)
            except Exception:
                return _call_chat(prompt)
        finally:
            if api_semaphore is not None:
                api_semaphore.release()

    meta: Dict[str, Any] = {
        "used_fallback": False,
        "fallback_improved": False,
        "upload_fmt": upload_fmt,
        "upload_jpeg_quality": int(upload_jpeg_quality),
        "upload_max_side": upload_max_side,
        "upload_grayscale": bool(upload_grayscale),
        "upload_info": upload_info,
    }

    t_api0 = time.perf_counter()
    raw = _call(OCR_MD_PROMPT)
    t_api1 = time.perf_counter()
    meta["t_api_primary_s"] = t_api1 - t_api0

    cleaned, flags = postprocess_and_assess(raw, boilerplate_patterns)
    meta.update(flags)
    primary_md, primary_flags = cleaned, dict(flags)

    if primary_flags.get("bad"):
        meta["used_fallback"] = True
        t_f0 = time.perf_counter()
        raw2 = _call(OCR_MD_PROMPT_FALLBACK)
        t_f1 = time.perf_counter()
        meta["t_api_fallback_s"] = t_f1 - t_f0

        cleaned2, flags2 = postprocess_and_assess(raw2, boilerplate_patterns)
        best_md, best_flags, picked_fallback = pick_better_candidate(
            primary_md, primary_flags, cleaned2, flags2
        )
        cleaned = best_md
        meta.update(best_flags)
        meta["fallback_improved"] = bool(picked_fallback)
        meta["fallback_primary_score"] = score_candidate(primary_md, primary_flags)
        meta["fallback_second_score"] = score_candidate(cleaned2, flags2)

    return cleaned.strip(), meta


# -------- main --------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=str, help="Input PDF")
    ap.add_argument("out_md", type=str, help="Output Markdown file")

    ap.add_argument("--max-tokens", type=int, default=None, help="Override settings.OCR_MAX_TOKENS (default: use settings)")
    ap.add_argument("--workers", type=int, default=None, help="Override settings.OCR_WORKERS (default: use settings)")
    ap.add_argument("--debug", action="store_true", help="Override settings.OCR_DEBUG=True for this run")
    ap.add_argument("--no-debug", action="store_true", help="Override settings.OCR_DEBUG=False for this run")
    args = ap.parse_args()

    cfg = load_config()
    api_key = require_str(cfg, "api_key")
    base_url = get_cfg(cfg, "base_url", "https://aihubmix.com/v1", expected_type=str, nonempty=True)
    model = require_str(cfg, "model")

    settings = load_settings()

    # dpi moved to settings.json
    dpi = int(get_setting(settings, "OCR_DPI", 350))
    if dpi < 72 or dpi > 1200:
        raise ValueError(f"Invalid OCR_DPI in settings.json: {dpi} (expected 72..1200)")


    pdf_path = Path(args.pdf).expanduser().resolve()
    out_md = Path(args.out_md).expanduser().resolve()
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(2)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    # ---- settings ----
    workers = int(get_setting(settings, "OCR_WORKERS", 4))
    api_conc = int(get_setting(settings, "OCR_API_CONCURRENCY", max(1, min(4, workers))))
    timeout_s = int(get_setting(settings, "OCR_TIMEOUT", 120))
    max_tokens_setting = get_setting(settings, "OCR_MAX_TOKENS", None)

    debug_enabled = bool(get_setting(settings, "OCR_DEBUG", False))
    verbose_page_log = bool(get_setting(settings, "OCR_VERBOSE_PAGE_LOG", True))
    debug_jpeg_quality = int(get_setting(settings, "OCR_DEBUG_JPEG_QUALITY", 90))

    # padding (proportional)
    pad_top = float(get_setting(settings, "OCR_PAD_TOP", 0.015))
    pad_other = float(get_setting(settings, "OCR_PAD_OTHER", 0.004))
    pad_max_px = int(get_setting(settings, "OCR_PAD_MAX_PX", 80))

    # upload encoding controls
    upload_fmt = str(get_setting(settings, "OCR_UPLOAD_FMT", "JPEG")).upper()
    upload_jpeg_quality = int(get_setting(settings, "OCR_UPLOAD_JPEG_QUALITY", 85))
    upload_max_side_raw = get_setting(settings, "OCR_UPLOAD_MAX_SIDE", "auto")
    upload_grayscale = bool(get_setting(settings, "OCR_UPLOAD_GRAYSCALE", False))

    # boilerplate patterns (configurable)
    boilerplate_patterns = compile_boilerplate_patterns(
        get_setting(settings, "OCR_STRIP_PREFIX_PATTERNS", [])
    )
    normalize_subpart_solution = bool(get_setting(settings, "OCR_NORMALIZE_SUBPART_SOLUTION_BOUNDARIES", True))
    normalize_solution_heading = bool(get_setting(settings, "OCR_NORMALIZE_SOLUTION_HEADINGS", True))

    # ---- CLI overrides ----
    if args.workers is not None:
        workers = int(args.workers)
    if workers <= 0:
        workers = 4

    if args.max_tokens is not None:
        ocr_max_tokens = int(args.max_tokens)
    else:
        ocr_max_tokens = max_tokens_setting
        if ocr_max_tokens is not None:
            ocr_max_tokens = int(ocr_max_tokens)

    if args.debug:
        debug_enabled = True
    if args.no_debug:
        debug_enabled = False

    api_conc = max(1, int(api_conc))
    if api_conc > workers:
        # allow, but usually you want api_conc <= workers
        pass

    # resolve upload_max_side (dpi-dynamic "auto")
    upload_max_side: Optional[int]
    if upload_max_side_raw is None:
        upload_max_side = None
    elif isinstance(upload_max_side_raw, str) and upload_max_side_raw.strip().lower() == "auto":
        upload_max_side = _default_upload_max_side_for_dpi(dpi)
    else:
        try:
            upload_max_side = int(upload_max_side_raw)  # may raise
        except Exception:
            upload_max_side = _default_upload_max_side_for_dpi(dpi)

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
    api_sem = threading.Semaphore(api_conc)

    # Debug dirs
    debug_dir = out_md.parent / f"{out_md.stem}_debug"
    pages_dir = debug_dir / "pages"
    per_page_md_dir = debug_dir / "page_md"

    # Page count
    t0 = time.perf_counter()
    n_pages = get_pdf_page_count(pdf_path)
    t1 = time.perf_counter()
    print(f"[init] pdf={pdf_path.name} pages={n_pages} dpi={dpi} (pdfinfo {t1 - t0:.3f}s)")
    print(
        f"[run] model={model} workers={workers} api_concurrency={api_conc} timeout={timeout_s}s "
        f"upload={upload_fmt} q={upload_jpeg_quality} max_side={upload_max_side} gray={upload_grayscale} "
        f"pad_top={pad_top} pad_other={pad_other} pad_max_px={pad_max_px} debug={debug_enabled}"
    )

    results: Dict[int, str] = {}
    quality: Dict[int, Dict[str, Any]] = {}

    def process_page(page_i: int) -> Tuple[int, str, Dict[str, Any]]:
        t_page0 = time.perf_counter()

        # 1) render single page (Standard Quality)
        t_r0 = time.perf_counter()
        raw_img = render_single_page(pdf_path, dpi=dpi, page_1based=page_i)
        t_r1 = time.perf_counter()

        # 2) padding
        ocr_img = pad_image(raw_img, pad_top=pad_top, pad_other=pad_other, max_px=pad_max_px)

        # 3) debug save (jpeg only)
        if debug_enabled:
            save_debug_images(pages_dir, page_i, raw_img, ocr_img, tag=f"dpi{dpi}", jpeg_quality=debug_jpeg_quality)

        # 4) OCR (First Attempt)
        t_o0 = time.perf_counter()
        md, meta = ocr_image_to_markdown(
            client,
            model,
            ocr_img,
            boilerplate_patterns,
            max_tokens=ocr_max_tokens,
            upload_fmt=upload_fmt,
            upload_jpeg_quality=upload_jpeg_quality,
            upload_max_side=upload_max_side,
            upload_grayscale=upload_grayscale,
            api_semaphore=api_sem,
            base_url_hint=base_url,
        )
        t_o1 = time.perf_counter()

        if meta.get("bad"):
            high_dpi = max(dpi + 150, 450) 
            tqdm.write(f"[Retry] Page {page_i} bad quality ({meta.get('bad_reason')}). Retrying at DPI {high_dpi}...")

            try:
                raw_img_hq = render_single_page(pdf_path, dpi=high_dpi, page_1based=page_i)
                ocr_img_hq = pad_image(raw_img_hq, pad_top=pad_top, pad_other=pad_other, max_px=pad_max_px)

                if debug_enabled:
                    save_debug_images(pages_dir, page_i, raw_img_hq, ocr_img_hq, tag=f"dpi{high_dpi}_RETRY", jpeg_quality=debug_jpeg_quality)

                md_retry, meta_retry = ocr_image_to_markdown(
                    client,
                    model,
                    ocr_img_hq,
                    boilerplate_patterns,
                    max_tokens=ocr_max_tokens,
                    upload_fmt=upload_fmt,
                    upload_jpeg_quality=upload_jpeg_quality,
                    upload_max_side=_default_upload_max_side_for_dpi(high_dpi),
                    upload_grayscale=upload_grayscale,
                    api_semaphore=api_sem,
                    base_url_hint=base_url,
                )

                first_md = md
                first_meta = dict(meta)
                first_score = score_candidate(first_md, first_meta)
                retry_score = score_candidate(md_retry, meta_retry)

                best_md, best_flags, picked_retry = pick_better_candidate(
                    first_md,
                    first_meta,
                    md_retry,
                    meta_retry,
                )
                md = best_md
                if picked_retry:
                    meta_retry["is_retry"] = True
                    meta_retry["retry_reason"] = first_meta.get("bad_reason")
                    meta_retry["retry_improved"] = True
                    meta_retry["retry_primary_score"] = first_score
                    meta_retry["retry_second_score"] = retry_score
                    meta = meta_retry
                else:
                    meta["is_retry"] = True
                    meta["retry_reason"] = first_meta.get("bad_reason")
                    meta["retry_improved"] = False
                    meta["retry_primary_score"] = first_score
                    meta["retry_second_score"] = retry_score
                
            except Exception as e:
                tqdm.write(f"[Retry Failed] Page {page_i}: {e}")
        t_page1 = time.perf_counter()

        out_meta: Dict[str, Any] = {
            "page": page_i,
            "dpi": dpi,
            "t_render_s": t_r1 - t_r0,
            "t_ocr_total_s": t_o1 - t_o0,
            "t_page_total_s": t_page1 - t_page0,
            "md_len": len(md),
            "full_meta": meta,
            "upload": meta.get("upload_info", {}),
        }

        return page_i, md, out_meta

    # Run with proper tqdm (no broken redraw). Use tqdm.write for per-page logs.
    page_indices = list(range(1, n_pages + 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_page, i): i for i in page_indices}
        with tqdm(total=n_pages, desc="OCR pages", dynamic_ncols=True) as pbar:
            for fut in as_completed(futs):
                page_i, md, meta = fut.result()
                results[page_i] = md
                quality[page_i] = meta
                pbar.update(1)

                if verbose_page_log:
                    fm = meta.get("full_meta") or {}
                    bad = bool(fm.get("bad"))
                    layout = bool(fm.get("layout"))
                    used_fallback = bool(fm.get("used_fallback"))
                    up = meta.get("upload") or {}
                    up_bytes = int(up.get("upload_bytes") or 0)
                    up_kb = up_bytes / 1024.0
                    up_wh = f"{up.get('upload_w')}x{up.get('upload_h')}"
                    orig_wh = f"{up.get('orig_w')}x{up.get('orig_h')}"
                    up_fmt2 = up.get("upload_fmt")
                    up_q2 = up.get("jpeg_quality")
                    up_gray2 = up.get("grayscale")

                    # tqdm.write(
                    #     f"[OCR] page={page_i} bad={bad} layout={layout} fallback={used_fallback} len={len(md)} "
                    #     f"render={meta.get('t_render_s', 0):.3f}s ocr={meta.get('t_ocr_total_s', 0):.3f}s "
                    #     f"upload={up_fmt2} q={up_q2} gray={up_gray2} orig={orig_wh} up={up_wh} bytes={up_kb:.1f}KB"
                    # )

    # Assemble output in order
    chunks: List[str] = []
    for i in range(1, n_pages + 1):
        md = normalize_exercise_bold_headers(results.get(i, ""))
        if normalize_solution_heading:
            md = normalize_solution_heading_lines(md)
        if normalize_subpart_solution:
            md = normalize_subpart_solution_boundaries(md)
        chunks.append(f"<!-- PAGE {i} -->\n{md}\n")
    out_md.write_text("\n".join(chunks), encoding="utf-8")

    if debug_enabled:
        debug_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "pdf": str(pdf_path),
            "out_md": str(out_md),
            "dpi": dpi,
            "workers": workers,
            "api_concurrency": api_conc,
            "timeout_s": timeout_s,
            "model": model,
            "pad_top": pad_top,
            "pad_other": pad_other,
            "pad_max_px": pad_max_px,
            "upload_fmt": upload_fmt,
            "upload_jpeg_quality": upload_jpeg_quality,
            "upload_max_side": upload_max_side,
            "upload_grayscale": upload_grayscale,
            "strip_prefix_patterns": [rx.pattern for rx in boilerplate_patterns],
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
