#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Book-oriented Markdown to LaTeX conversion."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI


try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None


# =========================
# LLM Prompts
# =========================

TAG_RECOVERY_PROMPT = (
    "Align equation numbers for ONE page of ConvexOptimizationSolutionsManual.\n"
    "Inputs are for the same page:\n"
    "A) OCR Markdown\n"
    "B) PDF plain text\n"
    "\n"
    "Task: ONLY insert missing LaTeX \\\\tag{...} into matching DISPLAY equations.\n"
    "\n"
    "Hard rules:\n"
    "1) Keep text exactly as-is: no rewrite, no reorder, no paraphrase.\n"
    "2) Allowed edit is ONLY adding \\\\tag{...}.\n"
    "3) Never modify existing tags; never duplicate tags.\n"
    "4) If matching is uncertain, skip that equation.\n"
    "5) Output Markdown content only. No explanation, no fences.\n"
    "6) Never output any instruction text.\n"
    "\n"
    "<<<MD>>>\n"
)

LATEX_CONVERT_PROMPT = (
    "You are converting OCR Markdown from a convex optimization solutions manual to LaTeX.\n"
    "\n"
    "No commentary. No code fences.\n"
    "\n"
    "Placeholders:\n"
    "- Tokens like ZZZ_MATHBLOCK_0001_ZZZ are immutable placeholders.\n"
    "- Keep each token EXACTLY unchanged, on its own line, same order.\n"
    "- Do not move, wrap, or alter placeholders.\n"
    "\n"
    "Content rules:\n"
    "- Preserve wording, order, and paragraph boundaries.\n"
    "- Keep exercise numbers and subparts (a)(b)(c) exactly.\n"
    "- Do not summarize, expand, or duplicate lines.\n"
    "- Do not output instruction/prompt text.\n"
    "- Keep readable paragraph spacing; do not collapse distinct paragraphs.\n"
    "- Keep list/subpart structure clear (e.g., (a), (b), (c) on separate lines when present).\n"
    "- Preserve subpart markers exactly: (a), (b), (c), ...\n"
    "- Keep subpart markers at line start when present; do not merge them into previous prose.\n"
    "- If a subpart marker appears after sentence punctuation, keep it as a new line item.\n"
    "- Do not treat phrases like 'part (a)' inside a sentence as a new subpart marker.\n"
    "- Do not renumber, relabel, or reorder subparts.\n"
    "- Do not merge content across different subpart markers.\n"
    "- If multiple Solution markers appear nearby, preserve each adjacent subpart boundary as-is.\n"
    "- Treat standalone figure placeholders like [Figure] or [Figure: ...] as OCR noise; remove them.\n"
    "- Never let a [Figure] placeholder break paragraph/subpart continuity.\n"
    "\n"
    "Environment rules:\n"
    "- Do NOT create any theorem/proof/section/chapter environments.\n"
    "- Do NOT create labels.\n"
    "- Output only converted LaTeX body text for this chunk.\n"
    "\n"
    "Math rules:\n"
    "- Inline math uses $...$.\n"
    "- Display math uses \\\\[ ... \\\\] unless an existing \\\\tag requires equation/align-like env.\n"
    "- Keep existing \\\\tag only; never invent new \\\\tag.\n"
    "- Keep formulas faithful; do not alter meaning.\n"
    "- Keep display equations separated from surrounding prose by blank lines when naturally present.\n"
    "\n"
    "Markdown:\n"
)

# Solution split prompt (greedy solution blocks may include trailing non-solution text)
PROOF_SPLIT_PROMPT = (
    "You are given OCR Markdown that contains an explicit 'Solution.' marker.\n"
    "The chunk may include non-solution text after the solution ends.\n"
    "Treat 'Solution.', '**Solution.**', and 'Solution:' as the same marker.\n"
    "\n"
    "Placeholders like ZZZ_MATHBLOCK_0001_ZZZ are immutable:\n"
    "- Keep unchanged, same order, on their own lines.\n"
    "- Do not move or wrap placeholders.\n"
    "\n"
    "Task:\n"
    "1) Split the INPUT MARKDOWN into (solution part) and (trailing non-solution part).\n"
    "2) Keep text order unchanged.\n"
    "3) Output in EXACT format:\n"
    "<<<PROOF_MD>>>\n"
    "<Markdown belonging to solution body only>\n"
    "<<<REST>>>\n"
    "<Remaining Markdown after the solution ends; empty if none>\n"
    "\n"
    "Rules:\n"
    "- Output only the required two blocks; no commentary.\n"
    "- Keep wording and order; no rewrite.\n"
    "- Remove the literal heading 'Solution.' from the proof body.\n"
    "- Also remove equivalent markdown-styled heading forms like '**Solution.**'.\n"
    "- Keep subparts (a)(b)(c) exactly.\n"
    "- Preserve each subpart marker at line start when present.\n"
    "- If markers appear after sentence punctuation, keep them as separate line-start items.\n"
    "- Do not treat inline mentions like 'part (a)' as subpart boundaries.\n"
    "- Do not renumber, relabel, merge, or split subparts.\n"
    "- Do not invent any new text.\n"
    "- Preserve paragraph boundaries and blank-line separation.\n"
    "- Do not convert to LaTeX in this step; keep Markdown text only.\n"
    "- Treat standalone [Figure] or [Figure: ...] lines as OCR placeholders, not semantic content.\n"
    "- Do NOT use [Figure] placeholders as evidence that the solution ended.\n"
    "- If [Figure] appears between proof paragraphs, ignore/remove it and keep surrounding proof text contiguous.\n"
    "- Special handling A (interleaved multipart pattern, e.g., Exercise 2.17):\n"
    "  when text contains repeated '(b)/(c)/(d) statement' followed by 'Solution.' markers,\n"
    "  treat each '(part statement) + Solution + proof body' as an independent segment.\n"
    "  Keep those later segments in <<<REST>>> in original order.\n"
    "  In this pattern, <<<PROOF_MD>>> should contain only the first solution body (usually for part (a)).\n"
    "  Do not merge all later parts into part (a).\n"
    "- Special handling B (cross-page continuation):\n"
    "  if a new page begins with '(b)' after a prior-page 'Solution.', treat it as continuation of the same exercise.\n"
    "  Do not move it to <<<REST>>> unless there is clear exercise boundary text.\n"
    "- Ignore running-header artifacts (e.g., 'Exercises', chapter title lines) when deciding solution boundaries.\n"
    "- If the solution end is clear, move following text into <<<REST>>>.\n"
    "- Never output instruction text.\n"
    "\n"
    "Markdown:\n"
)

SUBPART_RESEGMENT_PROMPT = (
    "You are given Markdown for one exercise solution body.\n"
    "Task: re-segment subparts so markers like (a), (b), (c) appear at line start when appropriate.\n"
    "\n"
    "Hard rules:\n"
    "- Preserve wording, formulas, order, and paragraph boundaries.\n"
    "- Do not add, remove, summarize, or rewrite content.\n"
    "- Do not invent new math or text.\n"
    "- You may insert or remove line breaks only to expose true subpart boundaries.\n"
    "- Prefer conservative segmentation: only split when a marker clearly starts a new item.\n"
    "- Do NOT split on phrases like 'part (a)' inside a sentence.\n"
    "- Keep placeholders like ZZZ_MATHBLOCK_0001_ZZZ unchanged and in order.\n"
    "- Output Markdown only, no commentary.\n"
    "\n"
    "Expected part labels for this exercise: "
)

ENVS = ["thm", "proof"]


# =========================
# Config helpers
# =========================

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


def find_settings_json() -> Path:
    path = Path(__file__).resolve().with_name("settings.json")
    if not path.exists():
        raise FileNotFoundError(f"settings.json not found next to script: {path}")
    return path


def load_settings() -> Dict[str, Any]:
    path = find_settings_json()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def get_setting(settings: Dict[str, Any], key: str, default: Any) -> Any:
    return settings.get(key, default)


def project_root_from_config() -> Path:
    return find_config_json().parent


# =========================
# Markdown: page split + tag recovery
# =========================

PAGE_SPLIT_RE = re.compile(r"(?m)^\s*<!--\s*PAGE\s+(\d+)\s*-->\s*$")


def split_markdown_pages(md_text: str) -> List[Tuple[int, str]]:
    parts = PAGE_SPLIT_RE.split(md_text)
    out: List[Tuple[int, str]] = []
    i = 1
    while i + 1 < len(parts):
        page_num = int(parts[i])
        content = (parts[i + 1] or "").strip()
        out.append((page_num, content))
        i += 2
    if not out:
        out = [(1, (md_text or "").strip())]
    return out


def _normalize_for_compare(s: str) -> str:
    """
    For "tag recovery" safety checks: compare texts while ignoring whitespace noise.
    """
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    # remove tags (we allow tags to be newly inserted)
    s = re.sub(r"\\tag\{[^}]*\}", "", s)
    # normalize spaces
    lines = []
    for ln in s.splitlines():
        ln2 = re.sub(r"\s+", " ", ln.strip())
        lines.append(ln2)
    return "\n".join(lines).strip()


def _similar_enough(a: str, b: str, *, threshold: float = 0.995) -> bool:
    a2 = _normalize_for_compare(a)
    b2 = _normalize_for_compare(b)
    if a2 == b2:
        return True
    ratio = SequenceMatcher(None, a2, b2).ratio()
    return ratio >= threshold


def _truncate_pdf_text(s: str, max_chars: int = 12000) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    # keep head+tail (equation number often near the end of the line, but not always)
    head = s[: max_chars // 2]
    tail = s[-max_chars // 2 :]
    return head + "\n...\n" + tail


_CHAT_FORCE_STREAM: Optional[bool] = None
_CHAT_TOKEN_PARAM: Optional[str] = None


def _is_stream_required_error(err: Exception) -> bool:
    msg = str(err or "").lower()
    return "stream must be set to true" in msg


def _is_unsupported_param_error(err: Exception, param_name: str) -> bool:
    msg = str(err or "").lower()
    return (
        "unsupported parameter" in msg
        and param_name.lower() in msg
    )


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


def _chat_completion_text(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    global _CHAT_FORCE_STREAM
    global _CHAT_TOKEN_PARAM

    base_kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
    )

    token_order = ["max_completion_tokens", "max_tokens"]
    if _CHAT_TOKEN_PARAM in token_order:
        token_order = [_CHAT_TOKEN_PARAM] + [x for x in token_order if x != _CHAT_TOKEN_PARAM]

    def _call(stream_mode: bool) -> str:
        global _CHAT_TOKEN_PARAM
        last_err: Optional[Exception] = None

        for token_param in token_order:
            kwargs = dict(base_kwargs)
            kwargs[token_param] = max_tokens
            try:
                if stream_mode:
                    stream_obj = client.chat.completions.create(stream=True, **kwargs)
                    text = _collect_stream_text(stream_obj).strip()
                else:
                    resp = client.chat.completions.create(**kwargs)
                    text = (resp.choices[0].message.content or "").strip()
                _CHAT_TOKEN_PARAM = token_param
                return text
            except Exception as e:
                last_err = e
                if _is_unsupported_param_error(e, token_param):
                    continue
                raise

        if last_err is not None:
            raise last_err
        return ""

    if _CHAT_FORCE_STREAM is True:
        return _call(stream_mode=True)

    try:
        return _call(stream_mode=False)
    except Exception as e:
        if _is_stream_required_error(e):
            _CHAT_FORCE_STREAM = True
            return _call(stream_mode=True)

        # Some gateways may return empty/non-JSON bodies on non-stream requests.
        # Try stream path once before surfacing the original exception.
        try:
            _CHAT_FORCE_STREAM = True
            return _call(stream_mode=True)
        except Exception:
            raise e


def infer_pdf_path(in_md: Path, pdf_arg: Optional[str]) -> Optional[Path]:
    """
    Try to find the corresponding PDF for tag recovery.

    Priority:
    1) --pdf argument
    2) infer from project structure:
         <root>/work/<rel>/<stem>/<stem>.md  ->  <root>/input_pdfs/<rel>/<stem>.pdf
    3) fallback: search near in_md for <stem>.pdf
    """
    if pdf_arg:
        p = Path(pdf_arg).expanduser()
        if p.exists():
            return p.resolve()

    root = project_root_from_config()
    work_dir = root / "work"
    input_dir = root / "input_pdfs"

    try:
        rel = in_md.resolve().relative_to(work_dir.resolve())
        stem = in_md.stem

        if len(rel.parts) >= 2 and rel.parts[-2] == stem and rel.parts[-1] == f"{stem}.md":
            pdf_rel = Path(*rel.parts[:-2]) / f"{stem}.pdf"
            cand = input_dir / pdf_rel
            if cand.exists():
                return cand.resolve()
    except Exception:
        pass

    # last resort: local search
    cand2 = in_md.with_suffix(".pdf")
    if cand2.exists():
        return cand2.resolve()
    cand3 = in_md.parent / f"{in_md.stem}.pdf"
    if cand3.exists():
        return cand3.resolve()

    return None


def extract_pdf_page_text(pdf_path: Path, page_1based: int) -> str:
    if fitz is None:
        return ""
    try:
        doc = fitz.open(pdf_path)
        idx = page_1based - 1
        if idx < 0 or idx >= doc.page_count:
            return ""
        page = doc.load_page(idx)
        txt = page.get_text("text") or ""
        doc.close()
        return txt.strip()
    except Exception:
        return ""


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=12),
    retry=retry_if_exception_type(Exception),
)
def llm_recover_page_tags(
    client: OpenAI,
    model: str,
    md_page: str,
    pdf_text: str,
    max_tokens: int,
) -> str:
    prompt = (
        TAG_RECOVERY_PROMPT
        + (md_page or "").strip()
        + "\n<<<PDF_TEXT>>>\n"
        + _truncate_pdf_text(pdf_text)
        + "\n"
    )
    return _chat_completion_text(
        client,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )


def pagewise_tag_recovery(
    pages: List[Tuple[int, str]],
    pdf_path: Optional[Path],
    client: OpenAI,
    tag_model: str,
    max_tokens_tag: int,
    workers: int,
) -> List[Tuple[int, str]]:
    """
    For each page, recover missing equation numbers by inserting \\tag{...}.
    If pdf_path is None or extraction fails, returns original pages unchanged.
    """
    if pdf_path is None or not pdf_path.exists():
        return pages

    # Extract all PDF page text first (IO-bound); keep in memory
    pdf_text_by_page: Dict[int, str] = {}
    for page_num, _md in pages:
        pdf_text_by_page[page_num] = extract_pdf_page_text(pdf_path, page_num)

    out: Dict[int, str] = {}

    def _process_one(page_num: int, md_page: str) -> Tuple[int, str]:
        pdf_text = pdf_text_by_page.get(page_num, "")
        if not md_page.strip() or not pdf_text.strip():
            return page_num, md_page

        fixed = llm_recover_page_tags(
            client=client,
            model=tag_model,
            md_page=md_page,
            pdf_text=pdf_text,
            max_tokens=max_tokens_tag,
        )

        # Safety gate: only accept if "mostly identical" ignoring added tags
        if fixed and _similar_enough(md_page, fixed):
            return page_num, fixed.strip()

        return page_num, md_page

    # Concurrency
    workers2 = max(1, int(workers or 1))
    with ThreadPoolExecutor(max_workers=workers2) as ex:
        futs = {ex.submit(_process_one, p, md): p for p, md in pages}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Tag recovery (page-wise)"):
            p, fixed = fut.result()
            out[p] = fixed

    return [(p, out.get(p, md)) for p, md in pages]


# =========================
# Heading injection (structure anchoring)
# =========================

HEADING_START = "<!-- HEADING_START -->"
HEADING_END = "<!-- HEADING_END -->"

MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SEC_LINE_RE = re.compile(r"^\s*(SECTION|CHAPTER)\s+(\d+)\s*$", re.IGNORECASE)
def _is_short_title_line(s: str) -> bool:
    s = s.strip()
    if not s or len(s) > 90:
        return False
    if s.endswith("."):
        return False
    if not re.search(r"[A-Za-z]", s):
        return False
    if re.search(r"\b(is|are|was|were|have|has|that|which|whenever)\b", s, re.IGNORECASE):
        return False
    return True


def _is_plain_heading_candidate(lines: List[str], idx: int) -> bool:
    raw = (lines[idx] or "").strip()
    # Normalize lightweight markdown wrappers before heading heuristics.
    s = raw.replace("**", "").replace("__", "").replace("`", "")
    s = re.sub(r"^\s*>+\s*", "", s).strip()
    if not _is_short_title_line(s):
        return False

    # Exclude structural markers handled elsewhere.
    if re.match(r"^\s*#+\s+", raw):
        return False
    if SEC_LINE_RE.match(s):
        return False
    if re.match(r"^\s*(?:Exercise\s+)?\d+(?:\.\d+)+\s*$", s, re.IGNORECASE):
        return False
    if re.match(r"^\s*Solution\s*[:.]?\s*$", s, re.IGNORECASE):
        return False
    if re.match(r"^\s*[-*]\s+", s):
        return False

    # Heading lines in this corpus are usually separated from surrounding prose.
    prev_blank = (idx == 0) or (not (lines[idx - 1] or "").strip())
    next_blank = (idx + 1 >= len(lines)) or (not (lines[idx + 1] or "").strip())
    if not (prev_blank or next_blank):
        return False

    words = re.findall(r"[A-Za-z][A-Za-z'-]*", s)
    if not words:
        return False
    # Avoid promoting short prose lead-ins (e.g., "The function") to headings.
    if len(words) <= 4 and words[0].lower() in {"the", "this", "that", "these", "those"}:
        return False
    # If next non-empty line is display math, current short line is often prose lead-in.
    j = idx + 1
    while j < len(lines) and not (lines[j] or "").strip():
        j += 1
    if j < len(lines):
        nxt = (lines[j] or "").strip()
        if re.match(r"^(?:\$\$|\\\[|\\begin\{(?:aligned|align\*?|equation\*?)\})", nxt):
            return False
    if len(words) > 12:
        return False
    caps = sum(1 for w in words if w[0].isupper())
    return (caps / len(words)) >= 0.5


def inject_heading_sentinels(full_md: str) -> str:
    """
    Scan merged Markdown and wrap headings with strong sentinels.

    Supported heading forms:
      - Markdown headings: #, ##, ...
      - "SECTION N" or "CHAPTER N" (optionally with a short title line right after)
    """
    lines = (full_md or "").splitlines()
    out: List[str] = []

    i = 0
    while i < len(lines):
        ln = lines[i].rstrip("\n")

        # (A) SECTION/CHAPTER blocks
        m = SEC_LINE_RE.match(ln.strip())
        if m:
            kind = m.group(1)
            num = m.group(2)

            # include optional next title line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            title_line = ""
            if j < len(lines) and _is_short_title_line(lines[j]):
                title_line = lines[j].rstrip("\n")

            out.append(HEADING_START)
            out.append(ln.strip())
            if title_line:
                out.append(title_line)
            out.append(HEADING_END)

            i = (j + 1) if title_line else (i + 1)
            continue

        # (B) Markdown headings
        hm = MD_HEADING_RE.match(ln.strip())
        if hm:
            out.append(HEADING_START)
            out.append(ln.strip())
            out.append(HEADING_END)
            i += 1
            continue

        # (C) Plain text headings preserved by upstream OCR prompt.
        if _is_plain_heading_candidate(lines, i):
            out.append(HEADING_START)
            out.append(ln.strip())
            out.append(HEADING_END)
            i += 1
            continue

        out.append(ln)
        i += 1

    return "\n".join(out).strip() + "\n"


# =========================
# Greedy chunking (logical chunking)
# =========================

_EXERCISE_START_RE = re.compile(
    r"^(?:(?P<prefix>Exercise)\s+)?(?P<num>\d+(?:\.\d+)+)\s*(?P<tail>.*)$",
    re.IGNORECASE,
)

_SOLUTION_START_RE = re.compile(
    # Be conservative: only treat explicit headings as proof starts.
    # Accept: "Solution", "Solution.", "Solution:", optionally with same-line tail after ./:.
    # Reject noun phrases like "Solution set ..." (problem statement content).
    r"^Solution(?:\s*$|\s*[:.]\s*(?P<tail>.*))$",
    re.IGNORECASE,
)


def _probe_structural_line(line: str) -> str:
    s = (line or "").strip()
    s = s.replace("**", "").replace("__", "").replace("`", "")
    s = re.sub(r"^\s*>+\s*", "", s)
    return s.strip()


def _normalize_exercise_line(line: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    s = _probe_structural_line(line)
    m = _EXERCISE_START_RE.match(s)
    if not m:
        return None, None, []

    num = (m.group("num") or "").strip()
    has_prefix = bool(m.group("prefix"))
    tail = (m.group("tail") or "").strip()
    if not num:
        return None, None, []

    # Allow "Exercise 2.1" without same-line tail; still reject bare "2.1" with empty tail.
    if not has_prefix and not tail:
        return None, None, []

    ex_id = f"Exercise {num}"
    lines = [ex_id]
    if tail:
        tail = tail.lstrip(":-–—. ").rstrip()
        if tail:
            lines.append(tail)
    return "thm", ex_id, lines


def _normalize_solution_line(line: str) -> Tuple[bool, List[str]]:
    s = _probe_structural_line(line)
    m = _SOLUTION_START_RE.match(s)
    if not m:
        return False, []

    tail = (m.group("tail") or "").strip()
    lines = ["Solution."]
    if tail:
        tail = tail.lstrip(":-–—. ").rstrip()
        if tail:
            lines.append(tail)
    return True, lines


@dataclass
class Block:
    kind: str              # "heading" | "exercise" | "proof" | "para"
    env: Optional[str]     # for exercise/proof: thm/proof
    md: str                # markdown payload


def greedy_chunk_markdown(anchored_md: str) -> List[Block]:
    """
    Streaming greedy scan.
    Priority:
      1) Heading sentinel blocks: forced break
      2) New environment starts (exercise/proof): forced break
      3) Proof blocks are greedy: they include everything until next heading or env-start
         (we do NOT try to detect end-of-proof here)
    """
    lines = (anchored_md or "").splitlines()

    blocks: List[Block] = []

    cur_kind: Optional[str] = None
    cur_env: Optional[str] = None
    cur_exercise_id: Optional[str] = None
    cur_lines: List[str] = []

    def flush() -> None:
        nonlocal cur_kind, cur_env, cur_exercise_id, cur_lines
        if cur_kind is None:
            return
        text = "\n".join(cur_lines).strip()
        if text:
            blocks.append(Block(kind=cur_kind, env=cur_env, md=text))
        cur_kind = None
        cur_env = None
        cur_exercise_id = None
        cur_lines = []

    i = 0
    while i < len(lines):
        ln = lines[i]

        # (A) Heading block
        if ln.strip() == HEADING_START:
            flush()
            i += 1
            heading_lines: List[str] = []
            while i < len(lines) and lines[i].strip() != HEADING_END:
                heading_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].strip() == HEADING_END:
                i += 1
            heading_text = "\n".join(heading_lines).strip()
            if heading_text:
                blocks.append(Block(kind="heading", env=None, md=heading_text))
            continue

        # (B) exercise start
        env, exercise_id, norm_stmt_lines = _normalize_exercise_line(ln)
        if env is not None:
            if cur_kind == "exercise" and cur_exercise_id and exercise_id == cur_exercise_id:
                # repeated title line across pages -> treat as continuation
                if len(norm_stmt_lines) > 1:
                    cur_lines.extend(norm_stmt_lines[1:])
                i += 1
                continue

            flush()
            cur_kind = "exercise"
            cur_env = env
            cur_exercise_id = exercise_id
            cur_lines = list(norm_stmt_lines)
            i += 1
            continue

        # (C) solution start
        is_proof, norm_proof_lines = _normalize_solution_line(ln)
        if is_proof:
            if cur_kind == "proof":
                # Keep repeated "Solution." markers inside proof blocks.
                # They are critical signals for multipart splitting downstream.
                cur_lines.extend(norm_proof_lines)
                i += 1
                continue

            flush()
            cur_kind = "proof"
            cur_env = "proof"
            cur_exercise_id = None
            cur_lines = list(norm_proof_lines)
            i += 1
            continue

        # (D) normal line
        if cur_kind is None:
            cur_kind = "para"
            cur_env = None
            cur_exercise_id = None
            cur_lines = [ln]
        else:
            cur_lines.append(ln)

        i += 1

    flush()
    return blocks


# =========================
# LaTeX cleanup / healing
# =========================

def strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^\s*```(?:latex)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def strip_outer_document(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"(?s)\\documentclass.*?\\begin\{document\}", "", s)
    s = re.sub(r"(?s)\\end\{document\}\s*$", "", s)
    return s.strip()


def strip_outer_env_wrapper(s: str, env: str) -> str:
    t = (s or "").strip()
    m = re.fullmatch(rf"(?s)\\begin\{{{re.escape(env)}\}}\s*(.*?)\s*\\end\{{{re.escape(env)}\}}", t)
    if m:
        return (m.group(1) or "").strip()
    return t


def normalize_display_math(latex: str) -> str:
    # $$...$$ -> \[...\]
    def repl(m: re.Match) -> str:
        inner = m.group(1).strip()
        return "\\[\n" + inner + "\n\\]"
    latex = re.sub(r"(?s)\$\$(.*?)\$\$", repl, latex)
    return latex


def normalize_unicode_symbols(latex: str) -> str:
    latex = latex.replace("§", r"\S ")
    latex = latex.replace("\u00A0", " ")
    return latex




# =========================
# Align/Tag fixes (unwrap illegal nesting + re-home misplaced \tag)
# =========================

# Bad pattern seen from OCR/LLM:
#   \[
#   \begin{aligned}
#   \begin{align*}
#      ...
#   \end{align*}
#   \end{aligned}
#   \tag{6.11}
#   \tag{6.12}
#   \]
#
# We must NOT keep align inside \[...\] (or inside aligned); it's illegal LaTeX.
# Also multiple trailing \tag lines MUST be attached to the corresponding align rows.
_WRAPPED_ALIGNED_ALIGN_RE = re.compile(
    r"(?s)"
    r"\\\[\s*"
    r"\\begin\{aligned\}\s*"
    r"(?P<align>\\begin\{align\*?\}.*?\\end\{align\*?\})\s*"
    r"\\end\{aligned\}\s*"
    r"(?P<tags>(?:\\tag\{[^}]+\}\s*)+)?"
    r"\\\]"
)

_WRAPPED_ALIGN_RE = re.compile(
    r"(?s)"
    r"\\\[\s*"
    r"(?P<align>\\begin\{align\*?\}.*?\\end\{align\*?\})\s*"
    r"(?P<tags>(?:\\tag\{[^}]+\}\s*)+)?"
    r"\\\]"
)


def _normalize_opt_operators(latex: str) -> str:
    """OCR often outputs non-standard commands like \minimize / \maximize."""
    latex = re.sub(r"\\minimize\b", r"\\min", latex)
    latex = re.sub(r"\\maximize\b", r"\\max", latex)
    return latex


def _distribute_tags_into_align_block(align_block: str, tags: List[str]) -> str:
    """
    Given a full align/align* environment, attach each tag to a suitable row.

    IMPORTANT:
    - Use a callable replacement for re.sub so backslashes in \tag are not
      interpreted as escapes (otherwise \t becomes a literal TAB).
    """
    if not tags:
        return align_block

    lines = align_block.splitlines()

    # locate begin/end lines
    try:
        begin_i = next(i for i, l in enumerate(lines) if re.search(r"\\begin\{align\*?\}", l))
    except StopIteration:
        return align_block
    try:
        end_i = max(i for i, l in enumerate(lines) if re.search(r"\\end\{align\*?\}", l))
    except ValueError:
        return align_block

    body = lines[begin_i + 1 : end_i]

    # candidate lines: prefer '=' constraints; fallback to <=/>= constraints;
    # avoid typical nonnegativity/all-quantifier tail
    eq_idxs: List[int] = []
    ineq_idxs: List[int] = []
    for i, ln in enumerate(body):
        if r"\tag" in ln:
            continue
        if "=" in ln:
            eq_idxs.append(i)
        if (r"\leq" in ln) or (r"\geq" in ln) or ("<" in ln) or (">" in ln):
            if re.search(r"\\geq\s*0", ln) and (r"\forall" in ln):
                continue
            ineq_idxs.append(i)

    if eq_idxs:
        cand = eq_idxs
    elif ineq_idxs:
        cand = ineq_idxs
    else:
        cand = list(range(len(body)))

    # choose targets: map to the last N candidate lines (tags usually belong to the tail constraints)
    if len(cand) >= len(tags):
        targets = cand[-len(tags) :]
    else:
        targets = cand + [cand[-1]] * (len(tags) - len(cand))

    for tag, bi in zip(tags, targets):
        ln = body[bi]
        if re.search(r"\\\\\s*$", ln):
            body[bi] = re.sub(
                r"(\\\\\s*)$",
                lambda m: f" {tag} {m.group(1)}",
                ln,
            ).rstrip()
        else:
            body[bi] = (ln.rstrip() + f" {tag}").rstrip()

    return "\n".join(lines[: begin_i + 1] + body + lines[end_i:])


def _fix_wrapped_align_blocks(latex: str) -> str:
    """
    Unwrap illegal wrappers like:
      - \[
        ... \begin{aligned}\begin{align*}...\end{align*}\end{aligned}
        ... \]
      - \[
        ... \begin{align*}...\end{align*}
        ... \]
    and re-home any trailing \tag{...} lines into the align rows.
    """

    def _repl(m: re.Match) -> str:
        align_block = m.group("align") or ""
        tags_text = m.group("tags") or ""
        tags = re.findall(r"\\tag\{[^}]+\}", tags_text)

        if tags:
            return _distribute_tags_into_align_block(align_block, tags)
        return align_block

    latex = _WRAPPED_ALIGNED_ALIGN_RE.sub(_repl, latex)
    latex = _WRAPPED_ALIGN_RE.sub(_repl, latex)
    return latex

def _unwrap_align_wrappers(latex: str) -> str:
    """
    Fix illegal nesting and misplaced tags around align/aligned blocks.

    We intentionally unwrap patterns like:
      - \[ \begin{aligned}\begin{align*}...\end{align*}\end{aligned} ... \]
      - \[ \begin{align*}...\end{align*} ... \]
    and move trailing \\tag{...} lines into the corresponding align rows.
    """
    latex = _fix_wrapped_align_blocks(latex)

    # If 'aligned' wraps an 'align' env (illegal), drop the aligned wrapper.
    latex = re.sub(
        r"(?s)\\begin\{aligned\}\s*(\\begin\{align\*?\}.*?\\end\{align\*?\})\s*\\end\{aligned\}",
        r"\1",
        latex,
    )

    # If 'equation' wraps an align env, drop the outer equation wrapper.
    latex = re.sub(
        r"(?s)\\begin\{equation\*?\}\s*(.*?\\begin\{align\*?\}.*?\\end\{align\*?\}.*?)\s*\\end\{equation\*?\}",
        r"\1",
        latex,
    )
    return latex


# =========================
# Array / bracket repairs (rule-based)
# =========================

_LEFT_RIGHT_BRACKET_RE = re.compile(r"(?s)\\left\[(?P<body>.*?)\\right\]")

def _rewrite_left_right_brackets_with_ampersand(latex: str) -> str:
    """Repair invalid patterns like ``\left[ -1 & 1 & -2 \right]``.

    ``&`` is only legal inside an alignment/matrix environment. For simple row/column
    vectors that were emitted as ``\left[ ... & ... \right]``, we rewrite them into a
    ``bmatrix`` so LaTeX can compile.
    """

    def _repl(m: re.Match) -> str:
        body = m.group("body") or ""
        if "&" not in body:
            return m.group(0)
        # If it's already a proper env, leave it.
        if "\\begin{" in body or "\\end{" in body:
            return m.group(0)
        inner = body.strip()
        return r"\begin{bmatrix}" + inner + r"\end{bmatrix}"

    return _LEFT_RIGHT_BRACKET_RE.sub(_repl, latex or "")


_ARRAY_ENV_RE = re.compile(
    r"(?s)\\begin\{array\}\{(?P<spec>[^}]*)\}(?P<body>.*?)\\end\{array\}"
)

def _rewrite_optimization_arrays_to_aligned(latex: str) -> str:
    """Convert broken optimization-problem ``array`` blocks into ``aligned``.

    OCR/LLMs frequently emit things like ``\begin{array}{rcl}`` but then produce rows
    with 4-6 ``&`` alignment tabs, causing "Extra alignment tab" errors.

    Heuristic: If an ``array`` contains ``\text{minimize}``, ``\text{maximize}``, or
    ``\text{subject to}``, rewrite the whole array body into:

    ``\begin{aligned}
       \text{minimize}\quad & ... \\
       \text{subject to}\quad & ... \\
       & ...
     \end{aligned}``
    """

    key_re = re.compile(r"\\text\{\s*(?:minimize|maximize|subject\s+to)\s*\}", re.I)

    def _row_to_aligned(row: str) -> str:
        row = (row or "").strip()
        if not row:
            return ""
        # drop horizontal rules (rare in this corpus)
        if row.strip() == r"\hline":
            return ""

        parts = [p.strip() for p in row.split("&")]

        # remove leading empties introduced by leading '&'
        while parts and parts[0] == "":
            parts = parts[1:]

        if not parts:
            return ""

        first = parts[0]
        rest = [p for p in parts[1:] if p]

        if key_re.search(first):
            expr = " ".join(rest).strip()
            return (first + r"\quad & " + expr).rstrip()

        expr = " ".join([p for p in parts if p]).strip()
        return ("& " + expr).rstrip()

    def _repl(m: re.Match) -> str:
        body = m.group("body") or ""
        if not key_re.search(body):
            return m.group(0)

        # Split rows on '\\' (optionally with spacing like '\\[6pt]')
        rows = re.split(r"\\\\(?:\[[^\]]*\])?", body)
        out_rows: List[str] = []
        for r in rows:
            aline = _row_to_aligned(r)
            if aline:
                out_rows.append(aline)

        if not out_rows:
            return m.group(0)

        return "\\begin{aligned}\n" + " \\\\\n".join(out_rows) + "\n\\end{aligned}"

    return _ARRAY_ENV_RE.sub(_repl, latex or "")



def _balance_delims(latex: str, left: str, right: str) -> str:
    """
    Ensure delimiter tokens are balanced by dropping unmatched rights and appending missing rights.
    Tokens must be simple string literals (no regex meta).
    """
    tok_re = re.compile(r"(" + re.escape(left) + r"|" + re.escape(right) + r")")
    out: List[str] = []
    idx = 0
    stack = 0
    for m in tok_re.finditer(latex):
        out.append(latex[idx:m.start()])
        tok = m.group(1)
        if tok == left:
            stack += 1
            out.append(tok)
        else:
            if stack > 0:
                stack -= 1
                out.append(tok)
            else:
                # drop unmatched right
                pass
        idx = m.end()
    out.append(latex[idx:])
    res = "".join(out)
    if stack > 0:
        res = res.rstrip() + "\n" + "\n".join([right] * stack) + "\n"
    return res


def _balance_display_math_delims(latex: str) -> str:
    """
    Balance display math delimiters '\\[' and '\\]' but ignore linebreak optional arguments like '\\[12pt]'.

    In LaTeX, '\\[<len>]' is part of the newline command '\\' (extra vertical space) and must NOT
    be treated as a display-math opener.
    """
    LEFT = r"\["
    RIGHT = r"\]"
    tok_re = re.compile(r"(?<!\\)(\\\[|\\\])")  # ignore '\\[' / '\\]'

    out: List[str] = []
    idx = 0
    stack = 0
    for m in tok_re.finditer(latex):
        out.append(latex[idx:m.start()])
        tok = m.group(1)
        if tok == LEFT:
            stack += 1
            out.append(tok)
        else:
            if stack > 0:
                stack -= 1
                out.append(tok)
            else:
                # drop unmatched right
                pass
        idx = m.end()
    out.append(latex[idx:])
    res = "".join(out)
    if stack > 0:
        res = res.rstrip() + "\n" + "\n".join([RIGHT] * stack) + "\n"
    return res


def _balance_inline_paren_math_by_paragraph(latex: str) -> str:
    """Balance ``\(...\)`` inline math *within each paragraph*.

    Why paragraph-level?
    - If we append a missing ``\)`` at the very end of a long document/block, a single
      missing closer (often from a hard line wrap) can accidentally pull *pages* of text
      into math mode.
    - In the OCR/PDF-to-text setting, inline math almost never intentionally spans across
      blank lines.

    Strategy:
    - Split by blank lines, balance tokens inside each paragraph.
    - Drop unmatched closers.
    - If a paragraph still has an unmatched opener, close it at the end of that paragraph.
    """

    s = latex or ""

    # Keep the blank-line separators so we preserve original spacing.
    parts = re.split(r"(\n\s*\n+)", s)
    out_parts: List[str] = []

    tok_re = re.compile(r"(\\\(|\\\))")

    for part in parts:
        # separator
        if re.fullmatch(r"\n\s*\n+", part or ""):
            out_parts.append(part)
            continue

        para = part or ""

        # Fix a frequent hard-wrap artifact: paragraph ends with an unterminated "\(X".
        m_tail = re.search(r"\\\(([A-Za-z]+)\s*$", para)
        if m_tail:
            var = m_tail.group(1)
            # If this paragraph already referenced var_k, reuse the max k (best-effort).
            subs = [int(x) for x in re.findall(rf"\\\({re.escape(var)}_(\\d+)\\\)", para)]
            if subs:
                para = re.sub(rf"\\\({re.escape(var)}\s*$", rf"\\({var}_{max(subs)}\\)", para)
            else:
                para = re.sub(rf"\\\({re.escape(var)}\s*$", rf"\\({var}\\)", para)

        # Balance tokens inside paragraph
        out: List[str] = []
        idx = 0
        stack = 0
        for m in tok_re.finditer(para):
            out.append(para[idx:m.start()])
            tok = m.group(1)
            if tok == r"\(":
                stack += 1
                out.append(tok)
            else:
                if stack > 0:
                    stack -= 1
                    out.append(tok)
                else:
                    # drop unmatched \)
                    pass
            idx = m.end()
        out.append(para[idx:])

        if stack > 0:
            out.append(r"\)" * stack)

        out_parts.append("".join(out))

    return "".join(out_parts)

def _balance_inline_dollars(latex: str) -> str:
    """Balance unescaped single ``$`` delimiters *within each paragraph*.

    Why paragraph-level?
    - In PDF/Markdown conversion, inline math almost never intentionally spans blank lines.
      If we balance only *globally*, a stray ``$`` in a later paragraph can "close" an opener
      from an earlier paragraph, swallowing a large span of prose into math mode and creating
      cascading errors.

    Rules (heuristic, conservative):
    - Ignore escaped ``\$`` and double ``$$``.
    - Split by blank lines (keep separators).
    - For each paragraph with an odd number of single ``$``:
        * If the last ``$`` sits at the paragraph end and looks stray (preceded by punctuation/space),
          drop it.
        * Otherwise, append a closing ``$`` at the paragraph end (before trailing whitespace).
    """

    def _single_dollar_positions(s: str) -> List[int]:
        pos: List[int] = []
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == "\\":  # skip escaped char
                i += 2
                continue
            if ch == "$":
                # ignore $$ (display math)
                if i + 1 < len(s) and s[i + 1] == "$":
                    i += 2
                    continue
                pos.append(i)
            i += 1
        return pos

    def _drop_char_at(s: str, k: int) -> str:
        return s[:k] + s[k + 1 :]

    # Split while keeping blank-line separators
    chunks = re.split(r"(\n[ \t]*\n+)", latex)
    out_chunks: List[str] = []

    for chunk in chunks:
        if re.fullmatch(r"\n[ \t]*\n+", chunk or ""):
            out_chunks.append(chunk)
            continue

        para = chunk
        dollars = _single_dollar_positions(para)
        if len(dollars) % 2 == 1:
            last = dollars[-1]
            tail = para[last + 1 :]
            prev = para[last - 1] if last > 0 else ""

            # If last $ is at paragraph end and looks like an accidental trailing symbol, drop it.
            if tail.strip() == "" and (prev.isspace() or prev in ".,;:!?)]}"):
                para = _drop_char_at(para, last)
            else:
                # Otherwise, close at end (before trailing whitespace)
                m = re.search(r"\s*$", para)
                if m:
                    para = para[: m.start()] + "$" + para[m.start() :]
                else:
                    para = para + "$"

        out_chunks.append(para)

    return "".join(out_chunks)
def _rewrite_tagged_displaymath(latex: str) -> str:
    """
    If we have \\[ ... \\tag{...} ... \\], rewrite to an equation environment,
    because \\tag is not valid inside \\[ ... \\] (displaymath).
    """
    def repl(m: re.Match) -> str:
        body = (m.group(1) or "").strip()
        if "\\tag{" not in body:
            return m.group(0)
        # If body already contains align-like env, keep (unwrap will handle separately)
        if re.search(r"\\begin\{align\*?\}", body):
            return m.group(0)
        return "\\begin{equation}\n" + body + "\n\\end{equation}"

    return re.sub(r"(?s)(?<!\\)\\\[\s*(.*?)\s*(?<!\\)\\\]", repl, latex)




def _rewrite_tagged_starred_math_envs(latex: str) -> str:
    """
    Fix cases like:
      \\begin{equation*} ... \\tag{...} ... \\end{equation*}
    by rewriting the environment to its non-starred variant, because \\tag is illegal in starred envs.
    """
    for base in ["equation", "align", "gather", "multline", "flalign"]:
        pat = re.compile(rf"(?s)\\begin\{{{base}\*\}}(.*?)\\end\{{{base}\*\}}")
        def _repl(m: re.Match) -> str:
            body = m.group(1) or ""
            if "\\tag{" not in body and "\\tag*{" not in body:
                return m.group(0)
            return rf"\begin{{{base}}}" + "\n" + body.strip("\n") + "\n" + rf"\end{{{base}}}"
        latex = pat.sub(_repl, latex)
    return latex


# ---- NEW: balance equation-like environments begin/end (and prevent nesting) ----
_MATH_ENV_TOKEN_RE = re.compile(
    r"^\s*\\(?P<kind>begin|end)\{(?P<env>(?:equation|align|gather|multline|flalign)\*?)\}\s*$"
)

INNER_END_RE = re.compile(
    r"\\end\{(?:array|cases|aligned|alignedat|split|matrix|pmatrix|bmatrix|vmatrix|Vmatrix|smallmatrix)\}\s*$"
)

def _looks_like_math_block(s: str) -> bool:
    """
    Heuristic: decide whether a small text block is likely "math" (used for conservative recovery).
    """
    if not s:
        return False

    # Strong signals
    if "\\tag{" in s or "\\tag*{" in s:
        return True
    if re.search(r"\\begin\{(?:array|cases|aligned|alignedat|split|matrix|pmatrix|bmatrix|vmatrix|Vmatrix|smallmatrix)\}", s):
        return True
    if "\\text{" in s:
        return True

    # Common math layout tokens
    if "&" in s and "\\\\" in s:
        return True

    # Common math macros (keep small; only those that almost never appear in prose)
    if re.search(r"\\(mathbf|mathbb|alpha|beta|gamma|delta)\b", s):
        return True

    # Weaker signals
    if re.search(r"\\(sum|int|frac|prod|sqrt|left|right|quad|cdot|leq|geq|neq|times|to|rightarrow|leftarrow)\b", s):
        return True
    if re.search(r"[=<>]", s) and re.search(r"\\[A-Za-z]+", s):
        return True
    return False



# ---- NEW: normalize accidental double-backslash \begin/\end ----
def normalize_double_backslash_begin_end(latex: str) -> str:
    """
    LLMs (or buggy string formatting) sometimes emit lines like:
        \\begin{equation}
        \\end{equation}
    which LaTeX interprets as a line-break (\\) followed by the text 'begin'/'end'.
    This function normalizes those to single-backslash commands.

    We fix both:
    - line-start (with optional indentation): ^\s*\\begin{...}
    - mid-line occurrences of exactly two backslashes not preceded by another backslash.
    """
    s = latex or ""
    # line start (allow indentation)
    s = re.sub(r"(?m)^(\s*)\\\\(begin|end)\{", r"\1\\\2{", s)
    # mid-line: exactly two backslashes, not preceded by another backslash
    s = re.sub(r"(?<!\\)\\\\(begin|end)\{", r"\\\1{", s)
    return s

def _is_hard_boundary(line: str) -> bool:
    s = (line or "").lstrip()
    if not s:
        return False
    if s.startswith(r"\section") or s.startswith(r"\subsection") or s.startswith(r"\subsubsection") or s.startswith(r"\chapter") or s.startswith(r"\paragraph"):
        return True
    if s.startswith(r"\clearpage") or s.startswith(r"\newpage"):
        return True
    if s.startswith("% ===== Page"):
        return True
    return False

def _balance_math_env_pairs(latex: str) -> str:
    """
    Heal common LLM mistakes for equation-like environments:
    - duplicate \\begin{equation} \\begin{equation}
    - missing \\end{equation} before a new \\begin{equation} (illegal nesting)
    - extra \\end{equation}
    - missing \\begin{equation} (we try to wrap the immediate preceding 'mathy' block)

    This is intentionally conservative and line-oriented.
    """
    lines = (latex or "").splitlines()
    out: List[str] = []
    stack: List[str] = []

    def prev_nonempty() -> str:
        for k in range(len(out) - 1, -1, -1):
            t = out[k].strip()
            if t:
                return t
        return ""

    def last_blank_pos() -> int:
        for k in range(len(out) - 1, -1, -1):
            if not out[k].strip():
                return k
        return -1

    for line in lines:
        # (0) If we are inside an equation-like env but hit a clear boundary (heading/page),
        #     close the env(s) first to avoid swallowing the rest of the document.
        if stack and _is_hard_boundary(line):
            while stack:
                out.append(rf"\\end{{{stack.pop()}}}")

        # (1) Common LLM bug: forget to close the outer equation-like env after finishing an inner env
        #     (array/aligned/cases/...). If the previous non-empty line is an inner \end{...}
        #     and the current line doesn't look like math, close the open env(s) here.
        if stack:
            prev = prev_nonempty()
            if prev and INNER_END_RE.search(prev):
                if line.strip() and not _MATH_ENV_TOKEN_RE.match(line) and not _looks_like_math_block(line):
                    while stack:
                        out.append(rf"\\end{{{stack.pop()}}}")

        m = _MATH_ENV_TOKEN_RE.match(line)
        if not m:
            out.append(line)
            continue

        kind = m.group("kind")
        env = m.group("env")

        if kind == "begin":
            if stack:
                # (A) duplicate begin -> drop
                if prev_nonempty() == rf"\begin{{{env}}}":
                    continue
                # (B) illegal nesting: close currently open env before starting a new one
                open_env = stack.pop()
                out.append(rf"\end{{{open_env}}}")
            stack.append(env)
            out.append(rf"\begin{{{env}}}")
            continue

        # kind == "end"
        if not stack:
            # (A) duplicate stray end right after another end -> drop
            prev = prev_nonempty()
            if prev.startswith(r"\end{equation") or prev.startswith(r"\end{align") or prev.startswith(r"\end{gather") or prev.startswith(r"\end{multline") or prev.startswith(r"\end{flalign"):
                continue

            # (B) try to recover a missing begin by wrapping the last short "mathy" block
            bpos = last_blank_pos()
            start = bpos + 1
            block_lines = out[start:]
            block = "\n".join(block_lines).strip("\n")

            if 1 <= len(block_lines) <= 10 and _looks_like_math_block(block):
                out.insert(start, rf"\begin{{{env}}}")
                out.append(rf"\end{{{env}}}")
            else:
                # give up: drop unmatched end to keep doc compilable
                continue
        else:
            open_env = stack.pop()
            # If mismatch (equation vs equation* etc), close what we opened.
            out.append(rf"\end{{{open_env}}}")

    # Close any remaining open envs
    while stack:
        out.append(rf"\end{{{stack.pop()}}}")

    return "\n".join(out)


# ---- NEW: tag filtering & de-duplication ----
_TAG_RE = re.compile(r"\\tag\*?\{([^}]+)\}")

def filter_and_dedupe_tags(latex: str, allowed_tags: set[str]) -> str:
    """
    1) If allowed_tags is non-empty: drop any \\tag{...} not in allowed_tags.
    2) Always de-duplicate: keep the FIRST occurrence of each tag and drop later repeats.
    """
    allowed = set([t.strip() for t in (allowed_tags or set()) if t and t.strip()])
    seen: set[str] = set()

    def _repl(m: re.Match) -> str:
        tag = (m.group(1) or "").strip()
        if allowed and tag not in allowed:
            return ""
        if tag in seen:
            return ""
        seen.add(tag)
        return m.group(0)

    return _TAG_RE.sub(_repl, latex or "")


def demote_untagged_numbered_math_envs(latex: str) -> str:
    """
    Convert numbered equation-like envs to starred versions when they have no explicit \\tag/\\label.
    This prevents auto-numbering that conflicts with textbook tags.
    """
    s = latex or ""
    for base in ["equation", "align", "gather", "multline", "flalign"]:
        pat = re.compile(rf"(?s)\\begin\{{{base}\}}(.*?)\\end\{{{base}\}}")
        def _repl(m: re.Match) -> str:
            body = m.group(1) or ""
            if "\\tag{" in body or "\\tag*{" in body or "\\label{" in body:
                return m.group(0)
            return rf"\begin{{{base}*}}" + "\n" + body.strip("\n") + "\n" + rf"\end{{{base}*}}"
        s = pat.sub(_repl, s)
    return s


# ---- NEW: figure placeholders (no images) ----
_FIG_RE = re.compile(r"(?s)\\begin\{figure\*?\}(?:\[[^\]]*\])?.*?\\end\{figure\*?\}")

def fix_missing_figures(latex: str, notice: str = r"\textit{[Image not available]}") -> str:
    """
    If the LLM outputs a figure environment with \\includegraphics but the image files do not exist,
    remove the \\includegraphics line(s), keep the caption, and insert a fixed short notice BEFORE the caption.
    """
    def _repl(m: re.Match) -> str:
        blk = m.group(0)
        lines = blk.splitlines()
        kept: List[str] = []
        for ln in lines:
            if "\\includegraphics" in ln:
                continue
            kept.append(ln)
        blk2 = "\n".join(kept)

        if notice in blk2:
            return blk2

        out_lines: List[str] = []
        inserted = False
        for ln in blk2.splitlines():
            if (not inserted) and re.search(r"\\caption", ln):
                out_lines.append(notice)
                inserted = True
            out_lines.append(ln)

        if not inserted:
            # no caption found; insert notice before the end
            blk3 = "\n".join(out_lines)
            blk3 = re.sub(r"(?m)^\\end\{figure\*?\}\s*$", notice + "\n\\end{figure}", blk3)
            return blk3

        return "\n".join(out_lines)

    return _FIG_RE.sub(_repl, latex or "")

# ---- NEW: wrap figure captions that are not in a figure environment ----
# Typical OCR output: "\textbf{Figure 6.2:} Caption text..."
_FIG_CAPTION_BOLD_RE = re.compile(
    r"^\s*\\textbf\{(?P<label>(?:Figure|Fig\.?)\s+\d+(?:\.\d+)*(?:\s*[:.]?)?)\}\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
_FIG_CAPTION_PLAIN_RE = re.compile(
    r"^\s*(?P<label>(?:Figure|Fig\.?)\s+\d+(?:\.\d+)*(?:\s*[:.]?)?)\s*(?P<rest>.*)$",
    re.IGNORECASE,
)


def _normalize_caption_text(label: str, rest: str) -> str:
    cap = ((label or "") + " " + (rest or "")).strip()
    # remove spaces before punctuation (very common OCR artifact)
    cap = re.sub(r"\s+([:;,\.])", r"\1", cap)
    cap = re.sub(r"\s{2,}", " ", cap)
    return cap.strip()


def _is_structural_begin_line(line: str) -> bool:
    """
    Lines that should never be swallowed by a preceding figure caption or example block.
    """
    s = (line or "").lstrip()
    if not s:
        return False
    if _is_hard_boundary(s):
        return True
    if re.match(r"^\\begin\{", s):
        return True
    if re.match(r"^%<BLOCK\b", s) or re.match(r"^%</BLOCK\b", s):
        return True
    return False


def wrap_figure_captions(latex: str, notice: str = r"\textit{[Image not available]}") -> str:
    """
    Many OCR'd textbooks have figure captions but no actual images.
    If we see a standalone caption line like:
        \\textbf{Figure 6.2:} ...
    we wrap it into:
        \\begin{figure}
        \\centering
        <notice>
        \\caption{...}
        \\end{figure}

    We only do this when NOT already inside a figure environment.
    """
    lines = (latex or "").splitlines()
    out: List[str] = []
    env_stack: List[str] = []

    def inside_figure() -> bool:
        return any(e.startswith("figure") for e in env_stack)

    def _push_env_from_line(s: str) -> None:
        mb = re.match(r"^\s*\\begin\{([^}]+)\}", s)
        if mb:
            env_stack.append(mb.group(1).strip())

    def _pop_env_from_line(s: str) -> None:
        me = re.match(r"^\s*\\end\{([^}]+)\}", s)
        if not me:
            return
        env = me.group(1).strip()
        # pop the last matching env if present; else ignore (keep doc compilable)
        for k in range(len(env_stack) - 1, -1, -1):
            if env_stack[k] == env:
                del env_stack[k:]
                return

    i = 0
    while i < len(lines):
        line = lines[i]

        if not inside_figure():
            m = _FIG_CAPTION_BOLD_RE.match(line)
            m_plain = None if m else _FIG_CAPTION_PLAIN_RE.match(line)

            # For plain (non-bold) lines, require a ":" to avoid wrapping prose like "Figure 6.1 illustrates ..."
            if m_plain and ":" not in (m_plain.group(0) or ""):
                m_plain = None

            if m or m_plain:
                mm = m or m_plain
                label = (mm.group("label") or "").strip()
                rest = (mm.group("rest") or "").strip()
                cap_lines: List[str] = []
                cap_lines.append(_normalize_caption_text(label, rest))

                j = i + 1
                while j < len(lines):
                    nxt = lines[j]
                    if not nxt.strip():
                        break
                    if _is_structural_begin_line(nxt):
                        break
                    # stop if another figure caption starts
                    if _FIG_CAPTION_BOLD_RE.match(nxt) or _FIG_CAPTION_PLAIN_RE.match(nxt):
                        break
                    cap_lines.append(nxt.strip())
                    j += 1

                caption = " ".join([c for c in cap_lines if c and c.strip()])
                caption = re.sub(r"\s{2,}", " ", caption).strip()

                out.append(r"\begin{figure}")
                out.append(r"\centering")
                out.append(notice)
                out.append(r"\caption{" + caption + r"}")
                out.append(r"\end{figure}")

                i = j
                continue

        # default: passthrough
        out.append(line)
        _push_env_from_line(line)
        _pop_env_from_line(line)
        i += 1

    return "\n".join(out)


def _heal_inner_env_counts(math_body: str, env: str) -> str:
    """
    Heal missing \\end{env} / extra \\end{env} inside a math body by simple counting.
    """
    b = len(re.findall(rf"\\begin\{{{re.escape(env)}\}}", math_body))
    e = len(re.findall(rf"\\end\{{{re.escape(env)}\}}", math_body))
    if b == e:
        return math_body
    if b > e:
        # append missing ends near the end
        return math_body.rstrip() + "\n" + ("\n".join([rf"\end{{{env}}}"] * (b - e))) + "\n"
    # too many ends: drop from the end
    diff = e - b
    for _ in range(diff):
        math_body = re.sub(rf"(?s)(.*)\\end\{{{re.escape(env)}\}}\s*$", r"\1", math_body, count=1)
    return math_body


def _heal_display_math_blocks(latex: str) -> str:
    """
    Heal common inner environments (array/cases/aligned/...) INSIDE display math blocks.

    Targets: array, cases, aligned, alignedat, split, matrix family.
    """
    targets = [
        "array",
        "cases",
        "aligned",
        "alignedat",
        "split",
        "matrix",
        "pmatrix",
        "bmatrix",
        "vmatrix",
        "Vmatrix",
        "smallmatrix",
    ]

    def heal_body(body: str) -> str:
        b = body
        for env in targets:
            b = _heal_inner_env_counts(b, env)
        return b

    # \[ ... \]
    def repl_display(m: re.Match) -> str:
        body = m.group(1) or ""
        body2 = heal_body(body)
        return "\\[\n" + body2.strip("\n") + "\n\\]"

    latex = re.sub(r"(?s)\\\[(.*?)\\\]", repl_display, latex)

    # equation/align-like env blocks
    for env_pat in [
        r"equation\*?",
        r"align\*?",
        r"gather\*?",
        r"multline\*?",
        r"flalign\*?",
    ]:
        def _repl_env(m: re.Match) -> str:
            envname = m.group(1)
            body = m.group(2) or ""
            body2 = heal_body(body)
            return rf"\begin{{{envname}}}" + "\n" + body2.strip("\n") + "\n" + rf"\end{{{envname}}}"

        latex = re.sub(
            rf"(?s)\\begin\{{({env_pat})\}}(.*?)\\end\{{\1\}}",
            _repl_env,
            latex,
        )

    return latex


def sanitize_latex_math(latex: str) -> str:
    latex = _unwrap_align_wrappers(latex)
    latex = _normalize_opt_operators(latex)
    # Repair optimization-problem arrays early (prevents '&' tab errors)
    latex = _rewrite_optimization_arrays_to_aligned(latex)
    # Repair bracketed vectors incorrectly written with '&'
    latex = _rewrite_left_right_brackets_with_ampersand(latex)
    latex = normalize_display_math(latex)
    latex = normalize_double_backslash_begin_end(latex)
    latex = _balance_display_math_delims(latex)
    # For inline \(...\), balance per-paragraph (avoid swallowing large spans of text)
    latex = _balance_inline_paren_math_by_paragraph(latex)
    latex = _balance_inline_dollars(latex)
    latex = _balance_math_env_pairs(latex)
    latex = _heal_display_math_blocks(latex)
    return latex.strip()


def heal_latex_fragment(latex: str) -> str:
    latex = strip_code_fences(latex)
    latex = strip_outer_document(latex)
    latex = normalize_unicode_symbols(latex)
    latex = sanitize_latex_math(latex)
    return latex.strip()



# =========================
# Post-fix: unwrap "prose mistakenly wrapped in display math"
# =========================

_PROSE_DM_CUE_RE = re.compile(
    r"(?i)\b(since|we have|we also have|it follows|indeed|finally|therefore|thus|hence|note that)\b"
)

def _split_by_text_macro(inner: str, macro: str = r"\text{") -> List[Tuple[str, str]]:
    """
    Split a math-mode string by occurrences of \text{...} while respecting (basic) brace nesting.

    Returns a list of ("math"|"text", segment). The sequence always starts with "math".
    """
    s = inner or ""
    out: List[Tuple[str, str]] = []
    i = 0
    L = len(macro)

    while True:
        j = s.find(macro, i)
        if j == -1:
            out.append(("math", s[i:]))
            break

        out.append(("math", s[i:j]))

        k = j + L
        depth = 1
        while k < len(s) and depth > 0:
            ch = s[k]
            prev = s[k - 1] if k > 0 else ""
            if ch == "{" and prev != "\\":
                depth += 1
            elif ch == "}" and prev != "\\":
                depth -= 1
            k += 1

        content = s[j + L : k - 1] if depth == 0 else s[j + L :]
        out.append(("text", content))
        i = k

    return out


def _wrap_inline_math(seg: str) -> str:
    """
    Wrap a segment in inline math $...$ if it looks mathy.
    Also detaches trailing punctuation .,;: outside the $...$ for nicer typography.
    """
    s = seg or ""
    if not s.strip():
        return s

    # If there's no obvious math token, leave it as plain text.
    if not re.search(r"[A-Za-z0-9\\_^=<>]|\\[a-zA-Z]+", s):
        return s

    m = s.strip()

    # Split trailing punctuation (avoid swallowing sentence punctuation inside math)
    trail = ""
    while m and m[-1] in ".,;:":
        trail = m[-1] + trail
        m = m[:-1].rstrip()

    if not m:
        return trail

    # Avoid double-wrapping (rare)
    if (m.startswith("$") and m.endswith("$")) or (m.startswith(r"\(") and m.endswith(r"\)")):
        return m + trail

    return f"${m}$" + trail


def _inline_from_segments(segs: List[Tuple[str, str]]) -> str:
    out: List[str] = []
    for kind, seg in segs:
        if kind == "text":
            out.append(seg)
        else:
            out.append(_wrap_inline_math(seg))
    return "".join(out)


def unwrap_prose_display_math(latex: str) -> str:
    """
    OCR sometimes mistakenly wraps an entire prose paragraph in display math,
    using many \text{...} fragments inside \[...\].

    Heuristic fix:
    - Only target \[...\] blocks that:
        * contain multiple \text{...}
        * do NOT contain line breaks (\\\\) or inner environments (\begin{...})
        * do NOT contain any \tag{...}
        * have substantial prose length inside \text{...}
    - Rewrite into:
        \[
          <first math segment>
        \]
        <rest as normal text with inline math>
    """
    s = latex or ""

    def _is_candidate(inner: str) -> bool:
        body = (inner or "").strip()
        if not body:
            return False
        if r"\tag{" in body or r"\tag*{" in body:
            return False
        if r"\begin{" in body or r"\end{" in body:
            return False
        if r"\\\\" in body:
            return False
        if body.count(r"\text{") < 2:
            return False

        segs = _split_by_text_macro(body)
        texts = [t.strip() for k, t in segs if k == "text" and t.strip()]
        total_text_len = sum(len(t) for t in texts)

        # Avoid touching short optimization keywords like "minimize"/"subject to"
        if total_text_len < 60:
            return False

        joined = " ".join(texts)
        if _PROSE_DM_CUE_RE.search(joined) or total_text_len >= 120:
            return True

        # Fallback: many words and punctuation usually indicates prose
        if len(joined.split()) >= 18 and joined.count(".") >= 1:
            return True

        return False

    def _repl(m: re.Match) -> str:
        inner = m.group(1) or ""
        if not _is_candidate(inner):
            return m.group(0)

        body = inner.strip()
        segs = _split_by_text_macro(body)
        if not segs:
            return m.group(0)

        # Keep the first math segment as a display equation (most textbooks intend that),
        # and convert the rest into normal prose with inline math.
        first_math = (segs[0][1] or "").strip()
        rest = _inline_from_segments(segs[1:]).strip()

        if not first_math:
            # No clear leading math -> fallback: inline all
            return _inline_from_segments(segs).strip()

        display = "\\[\n" + first_math + "\n\\]"
        if rest:
            return display + "\n" + rest
        return display

    # Apply to bracket display math only (the common OCR failure mode)
    return re.sub(r"(?s)\\\[(.*?)\\\]", _repl, s)



# =========================
# Display-math placeholders (preserve tags & boundaries)
# =========================

PLACEHOLDER_PREFIX = "ZZZ_MATHBLOCK_"
PLACEHOLDER_SUFFIX = "_ZZZ"

# Matches common display-math blocks in OCR Markdown / LaTeX fragments:
#   - $$ ... $$
#   - \[ ... \]
#   - \begin{equation/align/gather/multline/flalign[*]} ... \end{...}
_DISPLAY_MATH_BLOCK_RE = re.compile(
    r"(?s)"
    r"(?<!\\)\$\$.*?\$\$"
    r"|(?<!\\)\\\[.*?\\\]"
    r"|\\begin\{(?P<env>(?:equation|align|gather|multline|flalign)\*?)\}.*?\\end\{(?P=env)\}"
)

_PLACEHOLDER_RE = re.compile(r"\bZZZ_MATHBLOCK_\d{4}_ZZZ\b")
_TAG_TOKEN_RE = re.compile(r"\\tag\*?\{[^}]*\}")

# Manual textbook equation numbers often appear at display-math tail like:
#   ... \qquad (18.3)
# Convert them into canonical \tag{18.3}.
_TRAILING_MANUAL_EQNUM_RE = re.compile(
    r"(?s)^(?P<body>.*?)(?:(?:\s*(?:\\qquad|\\quad|\\hfill)\s*)+|\s+)\((?P<num>\d+(?:\.\d+)*(?:[A-Za-z])?)\)\s*$"
)

def _split_trailing_manual_eqnum(body: str) -> Tuple[str, Optional[str]]:
    s = (body or "").strip()
    if not s:
        return "", None
    m = _TRAILING_MANUAL_EQNUM_RE.match(s)
    if not m:
        return s, None
    core = (m.group("body") or "").rstrip()
    num = (m.group("num") or "").strip()
    if not num:
        return s, None
    return core, num

def _needs_aligned_wrapper(body: str) -> bool:
    s = (body or "")
    # Already has an inner structure that supports line breaks / alignment
    if re.search(r"\\begin\{(?:aligned|alignedat|split|array|cases|matrix|pmatrix|bmatrix|vmatrix|Vmatrix|smallmatrix)\}", s):
        return False
    # Needs structure if it contains linebreak/alignment tokens
    if "\\\\" in s or "&" in s:
        return True
    return False


def _build_bracket_display_math(inner: str) -> str:
    body = (inner or "").strip()
    original_body = body

    # Special case: \[\begin{aligned} ... \end{aligned}\] with per-row manual numbers
    # such as "... \qquad (18.55a)". Convert to align* with one \tag per row.
    m_al_rows = re.match(r"(?s)^\\begin\{aligned\}(?P<rows>.*)\\end\{aligned\}$", body)
    if m_al_rows:
        aligned_inner = (m_al_rows.group("rows") or "").strip()
        rows = [r.strip() for r in re.split(r"\\\\\s*", aligned_inner) if r.strip()]
        if rows:
            parsed_rows: List[Tuple[str, List[str]]] = []
            has_row_manual_num = False
            for row in rows:
                row_wo_manual, row_num = _split_trailing_manual_eqnum(row)
                row_tags = [t.strip() for t in _TAG_TOKEN_RE.findall(row_wo_manual) if t.strip()]
                row_core = _TAG_TOKEN_RE.sub("", row_wo_manual).strip()
                if row_num:
                    has_row_manual_num = True
                    manual_tag = rf"\\tag{{{row_num}}}"
                    if manual_tag not in row_tags and rf"\\tag*{{{row_num}}}" not in row_tags:
                        row_tags.append(manual_tag)

                # Deduplicate row tags while preserving order.
                seen_row = set()
                dedup_row_tags: List[str] = []
                for t in row_tags:
                    if t not in seen_row:
                        seen_row.add(t)
                        dedup_row_tags.append(t)

                parsed_rows.append((row_core, dedup_row_tags))

            if has_row_manual_num:
                out_rows: List[str] = []
                for row_core, row_tags in parsed_rows:
                    if row_tags:
                        out_rows.append((row_core + " " + row_tags[0]).strip())
                    else:
                        out_rows.append(row_core)
                return "\\begin{align*}\n" + " \\\\\n".join(out_rows) + "\n\\end{align*}"

    # Convert textbook-style trailing numbers like "\\qquad (18.3)" into a canonical \\tag{18.3}.
    body, trailing_num = _split_trailing_manual_eqnum(body)

    # Extract any \\tag tokens and move them OUTSIDE an eventual aligned wrapper
    tags = [t.strip() for t in _TAG_TOKEN_RE.findall(body) if t.strip()]
    body_wo_tags = _TAG_TOKEN_RE.sub("", body).strip()
    if trailing_num:
        manual_tag = rf"\\tag{{{trailing_num}}}"
        if manual_tag not in tags and rf"\\tag*{{{trailing_num}}}" not in tags:
            tags.append(manual_tag)

    # If a manual number remained before an extracted tag (e.g., "... \\qquad (18.20) \\tag{18.20}"),
    # remove the trailing manual number and keep canonical tag(s) only.
    body_wo_tags, trailing_num_2 = _split_trailing_manual_eqnum(body_wo_tags)
    if trailing_num_2:
        manual_tag_2 = rf"\\tag{{{trailing_num_2}}}"
        if manual_tag_2 not in tags and rf"\\tag*{{{trailing_num_2}}}" not in tags:
            tags.append(manual_tag_2)

    # Multiple tags are illegal in \[...\]. Convert to align* and attach one tag per row.
    if len(tags) > 1:
        core = body_wo_tags.strip()
        m_al = re.match(r"(?s)^\\begin\{aligned\}(.*?)\\end\{aligned\}$", core)
        if m_al:
            core = (m_al.group(1) or "").strip()
        rows = [r.strip() for r in re.split(r"\\\\\s*", core) if r.strip()]
        if rows:
            tagged_rows: List[str] = []
            for i, r in enumerate(rows):
                if i < len(tags):
                    tagged_rows.append((r + " " + tags[i]).strip())
                else:
                    tagged_rows.append(r)
            return "\\begin{align*}\n" + " \\\\\n".join(tagged_rows) + "\n\\end{align*}"
        # Fall back conservatively when rows cannot be split:
        # keep original content unchanged to avoid dropping manual numbering.
        return "\[\n" + original_body.strip("\n") + "\n\]"

    if _needs_aligned_wrapper(body_wo_tags):
        # Wrap multiline math in aligned
        body_wrapped = "\\begin{aligned}\n" + body_wo_tags.strip() + "\n\\end{aligned}"
    else:
        body_wrapped = body_wo_tags

    tag_text = ""
    if tags:
        tag_text = "\n" + "\n".join(tags)

    return "\\[\n" + body_wrapped.strip("\n") + tag_text + "\n\\]"


def sanitize_display_math_block(block: str) -> str:
    """
    Sanitize a display-math block extracted from Markdown.

    Goals:
      - Preserve the exact math content (do NOT rewrite symbols).
      - Ensure multiline display math is wrapped in an inner structure (aligned) when needed.
      - Ensure \tag{...} is NOT placed inside inner aligned/array/cases envs.
      - Normalize to \[ ... \] or align* for per-line tags.
    """
    b = (block or "").strip()

    # $$ ... $$  ->  \[ ... \]
    m = re.match(r"(?s)^\$\$(.*?)\$\$$", b)
    if m:
        return _build_bracket_display_math(m.group(1))

    # \[ ... \]
    m = re.match(r"(?s)^\\\[(.*?)\\\]$", b)
    if m:
        return _build_bracket_display_math(m.group(1))

    # equation/align-like envs
    m = re.match(
        r"(?s)^\\begin\{(?P<env>(?:equation|gather|multline|flalign|align)\*?)\}(?P<body>.*)\\end\{(?P=env)\}$",
        b,
    )
    if m:
        env = m.group("env")
        body = (m.group("body") or "").strip("\n")
        base = env[:-1] if env.endswith("*") else env

        if base == "align":
            # Use align* to suppress auto-numbering; keep per-line \tag{...} if present.
            return "\\begin{align*}\n" + body + "\n\\end{align*}"

        # Other equation-like envs -> normalize to \[ ... \]
        return _build_bracket_display_math(body)

    # Fallback: treat it as raw math and wrap
    return _build_bracket_display_math(b)


def normalize_manual_eqnums_to_tags_in_latex(latex: str) -> str:
    """
    Final-pass normalization on assembled LaTeX.
    Re-sanitize each display-math block so trailing textbook numbers like
    "... \qquad (18.3)" are converted into canonical "\tag{18.3}" when possible.
    """
    txt = latex or ""
    return _DISPLAY_MATH_BLOCK_RE.sub(lambda m: sanitize_display_math_block(m.group(0)), txt)


def replace_display_math_with_placeholders(markdown: str) -> Tuple[str, Dict[str, str], List[str]]:
    """
    Replace display-math blocks with stable placeholder tokens to prevent the LLM
    from moving math (and its \tag{...}) across the surrounding prose.

    Returns:
      - markdown_with_placeholders
      - placeholder->sanitized_math mapping
      - placeholder sequence in original order
    """
    mapping: Dict[str, str] = {}
    seq: List[str] = []
    idx = 0

    def _repl(m: re.Match) -> str:
        nonlocal idx
        idx += 1
        ph = f"{PLACEHOLDER_PREFIX}{idx:04d}{PLACEHOLDER_SUFFIX}"
        seq.append(ph)
        mapping[ph] = sanitize_display_math_block(m.group(0))
        # Put placeholders on their own line to reduce accidental editing.
        return "\n" + ph + "\n"

    md2 = _DISPLAY_MATH_BLOCK_RE.sub(_repl, markdown or "")
    return md2, mapping, seq


def restore_display_math_placeholders(latex: str, mapping: Dict[str, str]) -> str:
    """
    Restore sanitized display-math blocks back into LaTeX output.
    """
    if not mapping:
        return latex or ""

    def _repl(m: re.Match) -> str:
        token = m.group(0)
        return mapping.get(token, token)

    return _PLACEHOLDER_RE.sub(_repl, latex or "")


def split_markdown_by_display_math(markdown: str) -> List[Tuple[str, str]]:
    """
    Split markdown into [('text', ...), ('math', ...), ...] using _DISPLAY_MATH_BLOCK_RE.
    """
    s = markdown or ""
    out: List[Tuple[str, str]] = []
    pos = 0
    for m in _DISPLAY_MATH_BLOCK_RE.finditer(s):
        if m.start() > pos:
            out.append(("text", s[pos:m.start()]))
        out.append(("math", m.group(0)))
        pos = m.end()
    if pos < len(s):
        out.append(("text", s[pos:]))
    return out


# =========================
# Markdown post-fix: attach standalone equation numbers like "(6.4)"
# =========================

_STANDALONE_EQNUM_LINE_RE = re.compile(r"^\s*\((\d+(?:\.\d+)*(?:[A-Za-z])?)\)\s*$")

def _extract_display_math_inner(block: str) -> str:
    b = (block or "").strip()
    m = re.match(r"(?s)^\$\$(.*?)\$\$$", b)
    if m:
        inner = m.group(1)
    else:
        m = re.match(r"(?s)^\\\[(.*?)\\\]$", b)
        if m:
            inner = m.group(1)
        else:
            m = re.match(
                r"(?s)^\\begin\{(?P<env>(?:equation|gather|multline|flalign|align)\*?)\}(?P<body>.*)\\end\{(?P=env)\}$",
                b,
            )
            inner = (m.group("body") if m else b)

    inner = _TAG_TOKEN_RE.sub("", inner or "").strip()
    # If inner is already wrapped in aligned, unwrap for merging.
    m2 = re.match(r"(?s)^\\begin\{aligned\}(.*?)\\end\{aligned\}$", inner.strip())
    if m2:
        inner = m2.group(1).strip()
    return inner.strip()


def _merge_math_blocks_with_tag(math_blocks: List[str], tag_num: str) -> str:
    inners: List[str] = []
    for blk in math_blocks:
        inner = _extract_display_math_inner(blk)
        inner = re.sub(r"\\\\\s*$", "", inner).strip()
        if inner:
            inners.append(inner)

    if not inners:
        # degenerate
        return sanitize_display_math_block(math_blocks[-1]) if math_blocks else ""

    if len(inners) == 1:
        # Single display block: just add a tag.
        body = inners[0]
        tags = [rf"\tag{{{tag_num}}}"]
        # Wrap multiline if needed
        if _needs_aligned_wrapper(body):
            wrapped = "\\begin{aligned}\n" + body + "\n\\end{aligned}"
        else:
            wrapped = body
        return "\\[\n" + wrapped.strip("\n") + "\n" + "\n".join(tags) + "\n\\]"

    # Multiple blocks: merge into one aligned display with a single tag
    body = " \\\\\n".join(inners)
    return "\\[\n\\begin{aligned}\n" + body + "\n\\end{aligned}\n" + rf"\tag{{{tag_num}}}" + "\n\\]"


def attach_standalone_equation_numbers(markdown: str) -> str:
    """
    Convert standalone equation-number lines like '(6.4)' into \\tag{6.4} attached
    to the immediately preceding display math block(s).

    If multiple consecutive display-math blocks precede the number line, they are merged into a
    single aligned display equation carrying that tag.
    """
    segs = split_markdown_by_display_math(markdown)
    out: List[Tuple[str, str]] = []

    for kind, seg in segs:
        if kind != "text":
            out.append((kind, seg))
            continue

        # Process text line-by-line (keep linebreaks)
        for line in (seg or "").splitlines(True):
            m = _STANDALONE_EQNUM_LINE_RE.match(line.strip())
            if not m:
                out.append(("text", line))
                continue

            tag_num = m.group(1).strip()
            # Look back: collect trailing whitespace-only text segments
            while out and out[-1][0] == "text" and out[-1][1].strip() == "":
                out.pop()

            # Collect consecutive math blocks at the end (possibly separated by blank text)
            collected: List[str] = []
            while out and out[-1][0] == "math":
                collected.insert(0, out.pop()[1])
                while out and out[-1][0] == "text" and out[-1][1].strip() == "":
                    out.pop()

            if not collected:
                # No previous math -> keep as plain text
                out.append(("text", line))
                continue

            # If tag already exists in the collected math, just drop the standalone number line.
            if any(rf"\tag{{{tag_num}}}" in blk or rf"\tag*{{{tag_num}}}" in blk for blk in collected):
                for blk in collected:
                    out.append(("math", blk))
                continue

            merged = _merge_math_blocks_with_tag(collected, tag_num)
            if merged:
                out.append(("math", merged))

            # Drop the standalone line by not appending it.

    return "".join(seg for _, seg in out)


# =========================
# LaTeX post-fix: suppress auto-numbering for equation-like envs
# =========================

def star_all_equation_like_envs(latex: str) -> str:
    """
    Convert equation-like environments to their starred variants to suppress automatic numbering.
    Manual textbook numbering should be provided via \\tag{...}.
    """
    s = latex or ""
    for base in ["equation", "align", "gather", "multline", "flalign"]:
        s = re.sub(rf"\\begin\{{{base}\}}", rf"\\begin{{{base}*}}", s)
        s = re.sub(rf"\\end\{{{base}\}}", rf"\\end{{{base}*}}", s)
    return s

# =========================
# LLM calls
# =========================

class LowQualityLLMOutput(RuntimeError):
    """Raised when model output is clearly polluted or repetitively degenerate."""


_PROMPT_LEAK_PATTERNS = [
    r"(?i)the output must be latex",
    r"(?i)strict block rules",
    r"(?i)markdown:",
    r"(?i)you must keep every placeholder",
    r"(?i)output only latex",
]


def _has_prompt_leak(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return False
    return any(re.search(p, t) is not None for p in _PROMPT_LEAK_PATTERNS)


def _has_pathological_repetition(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return False

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) >= 20:
        # Too many identical lines usually means generation collapse.
        freq: Dict[str, int] = {}
        for ln in lines:
            freq[ln] = freq.get(ln, 0) + 1
        if max(freq.values()) >= 8:
            return True

    # Long numeric run like "1. 2. 3. ... 200."
    if re.search(r"(?:\b\d+\.\s*){80,}", t):
        return True

    # Repeated phrase loop (coarse heuristic).
    if re.search(r"(.{20,80})\1{4,}", re.sub(r"\s+", " ", t)):
        return True

    return False


_MD_LEAK_LINE_PATTERNS = [
    r"(?i)the output must be latex",
    r"(?i)output only latex",
    r"(?i)strict block rules",
    r"(?i)the input may contain placeholder tokens",
    r"(?i)you must keep every placeholder token exactly unchanged",
    r"(?i)do not move placeholders",
    r"(?i)critical:\s*do not invent any new \\tag",
    r"(?i)^markdown:\s*$",
    r"(?i)^math rules:\s*$",
    r"(?i)^formatting:\s*$",
    r"(?i)^task:\s*$",
    r"(?i)^hard rules",
    r"(?i)^<<<proof>>>$",
    r"(?i)^<<<rest>>>$",
]


def _strip_md_instruction_leakage(md: str) -> str:
    lines = (md or "").splitlines()
    out: List[str] = []
    for ln in lines:
        s = (ln or "").strip()
        if any(re.search(p, s) is not None for p in _MD_LEAK_LINE_PATTERNS):
            continue
        out.append(ln)
    return "\n".join(out).strip()


def _drop_runaway_number_lines(md: str) -> str:
    out: List[str] = []
    for ln in (md or "").splitlines():
        # pathological line like "1. 2. 3. ... 500."
        if re.search(r"(?:\b\d+\.\s*){80,}", ln):
            continue
        out.append(ln)
    return "\n".join(out).strip()


def _squash_repeated_lines(md: str, *, max_run: int = 2, min_len: int = 24) -> str:
    lines = (md or "").splitlines()
    out: List[str] = []
    prev_key = None
    run = 0

    for ln in lines:
        key = re.sub(r"\s+", " ", (ln or "").strip()).lower()
        comparable = len(key) >= min_len
        if comparable and key == prev_key:
            run += 1
        else:
            prev_key = key if comparable else None
            run = 1

        if comparable and run > max_run:
            continue
        out.append(ln)

    return "\n".join(out).strip()


_FIGURE_PLACEHOLDER_LINE_RE = re.compile(
    r"^\s*\[\s*figure(?:\s*:[^\]]*)?\s*\]\s*$",
    re.IGNORECASE,
)


def _drop_figure_placeholders(md: str) -> str:
    """
    Drop standalone OCR figure placeholders like:
      [Figure]
      [Figure: cone K with rays alpha, beta]
    Keep all normal prose/math lines unchanged.
    """
    lines = (md or "").splitlines()
    out: List[str] = []
    for ln in lines:
        if _FIGURE_PLACEHOLDER_LINE_RE.match(ln or ""):
            continue
        out.append(ln)

    # Avoid long blank gaps left by removed placeholders.
    s = "\n".join(out)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def sanitize_ocr_markdown(md: str) -> str:
    """
    Defensive pre-clean for OCR markdown:
    - strip leaked instruction lines from prompts
    - drop obvious runaway number-loop lines
    - squash long repeated line runs
    - drop standalone [Figure] placeholders (kept in OCR stage only)
    """
    s = (md or "").replace("\r\n", "\n").replace("\r", "\n")
    s = _strip_md_instruction_leakage(s)
    s = _drop_runaway_number_lines(s)
    s = _squash_repeated_lines(s, max_run=2, min_len=24)
    s = _drop_figure_placeholders(s)
    return s.strip()


def _validate_llm_tex_output(raw: str) -> None:
    if _has_prompt_leak(raw):
        raise LowQualityLLMOutput("LLM output leaked prompt text.")
    if _has_pathological_repetition(raw):
        raise LowQualityLLMOutput("LLM output is pathologically repetitive.")


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    retry=retry_if_exception_type(Exception),
)
def markdown_to_latex(client: OpenAI, model: str, markdown: str, max_tokens: int) -> str:
    """
    Convert OCR Markdown to LaTeX, while preserving display-math blocks verbatim (including \\tag{...}).

    Strategy:
      1) Replace display-math blocks with stable placeholders.
      2) Ask the LLM to convert the remaining text, keeping placeholders unchanged.
      3) Restore sanitized math blocks.
      4) Heal common LaTeX delimiter issues.
      5) If the LLM tampers with placeholders, fall back to segment-by-segment conversion.
    """
    md_in = sanitize_ocr_markdown(markdown or "")

    md_ph, mapping, seq = replace_display_math_with_placeholders(md_in)
    prompt = LATEX_CONVERT_PROMPT + md_ph.strip()

    raw = _chat_completion_text(
        client,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    raw = strip_code_fences(raw)
    raw = strip_outer_document(raw)
    _validate_llm_tex_output(raw)

    if mapping:
        found = _PLACEHOLDER_RE.findall(raw)
        ok = (found == seq) and (len(found) == len(set(found)))
        if not ok:
            # Fallback: convert non-math text segments separately and interleave with math blocks.
            parts: List[str] = []
            for kind, seg in split_markdown_by_display_math(md_in):
                if kind == "math":
                    parts.append(sanitize_display_math_block(seg))
                else:
                    t = (seg or "").strip()
                    if t:
                        parts.append(markdown_to_latex(client, model, t, max_tokens))
            merged = "\n\n".join([p for p in parts if p and p.strip()])
            return heal_latex_fragment(merged)

        raw = restore_display_math_placeholders(raw, mapping)

    return heal_latex_fragment(raw)

def _parse_proof_split_response(text: str) -> Tuple[str, str]:
    t = (text or "").strip()
    t = strip_code_fences(t)
    t = strip_outer_document(t)
    m = re.search(r"(?s)<<<PROOF_MD>>>\s*(.*?)\s*<<<REST>>>\s*(.*)\s*$", t)
    if not m:
        return "", ""
    return m.group(1).strip(), m.group(2).strip()


# Backward-compat alias: earlier iterations used this helper name.
# v5 accidentally called _split_proof_output() but only defined
# _parse_proof_split_response(). Keep both.
def _split_proof_output(text: str) -> Tuple[str, str]:
    """Parse the proof-split LLM output into (proof_part, rest_part)."""
    return _parse_proof_split_response(text)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    retry=retry_if_exception_type(Exception),
)
def markdown_proof_split_to_latex(
    client: OpenAI,
    model: str,
    markdown: str,
    max_tokens: int,
) -> Tuple[str, str]:
    """
    Ask the LLM to split OCR Markdown into:
      - solution Markdown body (without literal "Solution.")
      - trailing non-solution Markdown

    We preserve display-math placeholders and REQUIRE exact placeholder sequence match.
    """
    md_in = sanitize_ocr_markdown(markdown or "")
    md_ph, mapping, seq = replace_display_math_with_placeholders(md_in)

    prompt = PROOF_SPLIT_PROMPT + md_ph.strip()
    raw = _chat_completion_text(
        client,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    raw = strip_code_fences(raw)
    raw = strip_outer_document(raw)
    _validate_llm_tex_output(raw)

    proof_part, rest_part = _split_proof_output(raw)

    if mapping:
        found = _PLACEHOLDER_RE.findall(proof_part + "\n" + rest_part)
        ok = (found == seq) and (len(found) == len(set(found)))
        if not ok:
            # Strict fallback: keep whole chunk as solution body to avoid silent math loss/reorder.
            proof_part = md_in
            rest_part = ""
        proof_part = restore_display_math_placeholders(proof_part, mapping)
        rest_part = restore_display_math_placeholders(rest_part, mapping)

    return proof_part.strip(), rest_part.strip()


@retry(
    reraise=True,
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    retry=retry_if_exception_type(Exception),
)
def llm_resegment_subparts_markdown(
    client: OpenAI,
    model: str,
    exercise_num: str,
    expected_parts: List[str],
    proof_md: str,
    max_tokens: int,
) -> str:
    """
    Ask LLM to keep content unchanged but normalize subpart markers to line-start form.
    This is a repair pass used only when proof subparts are missing against statement parts.
    """
    md_in = sanitize_ocr_markdown(proof_md or "")
    if not md_in.strip() or not expected_parts:
        return md_in

    md_ph, mapping, seq = replace_display_math_with_placeholders(md_in)
    expected_txt = ", ".join([f"({p})" for p in expected_parts])
    prompt = (
        SUBPART_RESEGMENT_PROMPT
        + expected_txt
        + "\n"
        + f"Exercise: {exercise_num}\n"
        + "\nMarkdown:\n"
        + md_ph.strip()
    )
    raw = _chat_completion_text(
        client,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    ).strip()
    raw = strip_code_fences(raw)
    raw = strip_outer_document(raw)

    if mapping:
        found = _PLACEHOLDER_RE.findall(raw)
        ok = (found == seq) and (len(found) == len(set(found)))
        if not ok:
            return md_in
        raw = restore_display_math_placeholders(raw, mapping)

    return sanitize_ocr_markdown(raw)

# =========================
# Block sentinels (for texTojson_new)
# =========================

ENV_BLOCK_RE = re.compile(
    r"\\begin\{(?P<env>thm|proof)\}\s*(?P<body>.*?)\\end\{\1\}",
    re.DOTALL,
)

EXERCISE_TITLE_RE = re.compile(r"\bExercise\s+(\d+(?:\.\d+)+)\b", re.IGNORECASE)
_THEOREM_DEP_RE = re.compile(
    r"(?i)\b(?P<kind>Theorem|Thm\.?|Lemma|Lem\.?|Algorithm|Alg\.?)\s*(?P<num>\d+(?:\.\d+)*)\b"
)


def _theorem_refs_attr_from_text(text: str) -> str:
    seen = set()
    out = []
    for m in _THEOREM_DEP_RE.finditer(text or ""):
        kind_raw = (m.group("kind") or "").strip().lower().rstrip(".")
        num = (m.group("num") or "").strip().rstrip(".")
        if kind_raw.startswith("lem"):
            kind = "lemma"
        elif kind_raw.startswith("alg"):
            kind = "algorithm"
        else:
            kind = "theorem"
        dep = f"{kind}:{num}" if num else ""
        if not dep or dep in seen:
            continue
        seen.add(dep)
        out.append(dep)
    return ";".join(out)


def _first_nonempty_line(s: str) -> str:
    for ln in (s or "").splitlines():
        t = ln.strip()
        if t:
            return t
    return ""


def _strip_simple_latex_cmds(s: str) -> str:
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
    s = s.replace("\\(", "").replace("\\)", "")
    s = s.replace("\\[", "").replace("\\]", "")
    return s.strip()


def extract_short_label(env: str, body: str) -> str:
    if env == "proof":
        return "Proof"

    head = "\n".join((body or "").splitlines()[:8])
    head_plain = _strip_simple_latex_cmds(head)

    m = EXERCISE_TITLE_RE.search(head_plain)
    if m:
        return f"Exercise {m.group(1)}"

    line = _strip_simple_latex_cmds(_first_nonempty_line(body))
    if not line:
        return "UNKNOWN"
    if len(line) > 120:
        return line[:120] + "..."
    return line


def escape_attr(s: str) -> str:
    s = (s or "").replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    return s


def insert_block_sentinels(latex: str) -> str:
    if "%<BLOCK" in (latex or ""):
        return (latex or "").strip()

    out: List[str] = []
    pos = 0
    for m in ENV_BLOCK_RE.finditer(latex or ""):
        out.append((latex or "")[pos:m.start()])
        env = m.group("env")
        body = m.group("body")

        label = extract_short_label(env, body)
        refs_attr = ""
        if env == "thm" and label.lower().startswith("exercise "):
            refs_attr = _theorem_refs_attr_from_text(body)
        attr_tail = f' refs="{escape_attr(refs_attr)}"' if refs_attr else ""
        out.append(f'%<BLOCK type={env} label="{escape_attr(label)}"{attr_tail}>\n')
        out.append(m.group(0))
        out.append("\n%</BLOCK>\n")
        pos = m.end()

    out.append((latex or "")[pos:])
    return "".join(out).strip()


# =========================
# Heading -> LaTeX (deterministic)
# =========================

def heading_block_to_latex(heading_md: str) -> str:
    """
    Keep heading text as plain body text.
    We use heading blocks only as chunk boundaries for stability.
    """
    lines = [ln.strip() for ln in (heading_md or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        hm = MD_HEADING_RE.match(lines[0])
        if hm:
            return hm.group(2).strip()
        return lines[0]
    return "\n".join(lines).strip()




def split_large_para_blocks(blocks: List[Block], max_chars: int = 8000) -> List[Block]:
    """
    Prevent LLM truncation by splitting oversized 'para' blocks into smaller chunks.

    We split greedily on paragraph boundaries (blank lines). This preserves equations and list blocks
    reasonably well for OCR Markdown.
    """
    max_chars = max(1000, int(max_chars or 8000))
    out: List[Block] = []
    for blk in blocks:
        if blk.kind != "para" or len(blk.md) <= max_chars:
            out.append(blk)
            continue

        # Split on blank-line paragraph boundaries
        parts = re.split(r"\n\s*\n+", blk.md.strip())
        buf: List[str] = []
        cur = 0
        for part in parts:
            part = part.strip()
            if not part:
                continue
            add_len = len(part) + (2 if buf else 0)
            if buf and (cur + add_len) > max_chars:
                out.append(Block(kind="para", env=None, md="\n\n".join(buf).strip() + "\n"))
                buf = [part]
                cur = len(part)
            else:
                cur += add_len
                buf.append(part)

        if buf:
            out.append(Block(kind="para", env=None, md="\n\n".join(buf).strip() + "\n"))
    return out

# =========================
# Document template
# =========================

def build_tex_document(body_chunks: List[str]) -> str:
    preamble = r"""\documentclass[11pt]{book}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{microtype}
\geometry{margin=1in}

% OCR sometimes outputs non-standard optimization operators
\providecommand{\minimize}{\min}
\providecommand{\maximize}{\max}

% Unnumbered Exercise container for downstream block parsing.
\newtheorem*{thm}{Exercise}

% Readability-oriented spacing for OCR-converted text.
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5em}
\setlength{\abovedisplayskip}{8pt}
\setlength{\belowdisplayskip}{8pt}
\setlength{\abovedisplayshortskip}{6pt}
\setlength{\belowdisplayshortskip}{6pt}

\begin{document}
"""
    end = r"""
\end{document}
"""
    return preamble + "\n\n".join([c for c in body_chunks if c and c.strip()]).rstrip() + "\n" + end


def _split_first_nonempty(md: str) -> Tuple[str, str]:
    lines = (md or "").splitlines()
    first_idx = -1
    for i, ln in enumerate(lines):
        if ln.strip():
            first_idx = i
            break
    if first_idx < 0:
        return "", ""
    first = lines[first_idx].strip()
    rest = "\n".join(lines[first_idx + 1 :]).strip()
    return first, rest


def _wrap_env(env: str, body: str, lead_line: str = "") -> str:
    parts: List[str] = [rf"\begin{{{env}}}"]
    if lead_line.strip():
        parts.append(lead_line.strip())
    if body.strip():
        parts.append(body.strip())
    parts.append(rf"\end{{{env}}}")
    return ("\n\n" + "\n".join(parts).strip() + "\n\n").strip()


_SUBPART_LINE_RE = re.compile(
    r"^\s*(?:\*\*|__)?\(\s*(?P<part>[a-z])\s*\)(?:\*\*|__)?(?:[.:])?\s*(?P<tail>.*)$"
)
_EXERCISE_NUM_RE = re.compile(r"^\s*Exercise\s+(?P<num>\d+(?:\.\d+)*(?:[A-Za-z])?)\s*$", re.IGNORECASE)


def split_subparts(md: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Split markdown by line-start subpart markers like "(a) ...".
    Returns:
      - prelude text before first subpart marker
      - list of (part_id, subpart_markdown)
    """
    lines = (md or "").splitlines()
    marks: List[Tuple[int, str]] = []
    for i, ln in enumerate(lines):
        m = _SUBPART_LINE_RE.match((ln or "").strip())
        if m:
            marks.append((i, (m.group("part") or "").lower()))

    if not marks:
        return (md or "").strip(), []

    first_i = marks[0][0]
    # Keep text before the first "(a)" as a shared prelude.
    prelude = "\n".join(lines[:first_i]).strip()

    parts: List[Tuple[str, str]] = []
    for k, (start_i, part_id) in enumerate(marks):
        end_i = marks[k + 1][0] if (k + 1) < len(marks) else len(lines)
        seg = "\n".join(lines[start_i:end_i]).strip()
        if seg:
            parts.append((part_id, seg))
    return prelude, parts


def split_subparts_expected(
    md: str,
    expected_parts: List[str],
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Like split_subparts(), but only splits at the expected top-level part markers IN ORDER.
    Each expected part is matched at most ONCE, scanning forward from the previous match.
    Any sub-markers that appear INSIDE the last expected part (e.g., (a)(b)(c) steps
    inside part (e)'s proof) are treated as content, not as new top-level splits.

    Example: statement has parts (a,b,c,d,e); proof has (a)...(b)...(e): ...(a)...(b)...(c)...
    The trailing (a)(b)(c) after (e) are sub-steps of (e), so they go into (e)'s block.
    """
    if not expected_parts:
        return split_subparts(md)

    lines = (md or "").splitlines()
    mark_positions: List[Tuple[int, str]] = []
    search_from = 0

    for part_id in expected_parts:
        for i in range(search_from, len(lines)):
            m = _SUBPART_LINE_RE.match((lines[i] or "").strip())
            if m and (m.group("part") or "").lower() == part_id:
                mark_positions.append((i, part_id))
                search_from = i + 1
                break

    if not mark_positions:
        return (md or "").strip(), []

    first_i = mark_positions[0][0]
    prelude = "\n".join(lines[:first_i]).strip()

    parts: List[Tuple[str, str]] = []
    for k, (start_i, part_id) in enumerate(mark_positions):
        # End boundary is the NEXT expected marker, not any marker.
        end_i = mark_positions[k + 1][0] if (k + 1) < len(mark_positions) else len(lines)
        seg = "\n".join(lines[start_i:end_i]).strip()
        if seg:
            parts.append((part_id, seg))
    return prelude, parts


def _is_structural_boundary_md_line(line: str) -> bool:
    s = _probe_structural_line(line or "")
    if not s:
        return False
    if _EXERCISE_START_RE.match(s):
        return True
    if SEC_LINE_RE.match(s):
        return True
    if s.startswith("#"):
        return True
    if s in {HEADING_START, HEADING_END}:
        return True
    return False


def _extract_interleaved_subpart_statements_from_proof(md: str) -> List[Tuple[str, str]]:
    """
    Recover subpart statement lines from proof markdown when OCR interleaves:
      (b) <statement...>
      Solution.
      <proof...>
      (c) <statement...>
      Solution.
      ...

    Returns a list of (part_id, statement_markdown), preserving source order.
    """
    lines = (md or "").splitlines()
    out: List[Tuple[str, str]] = []
    seen: set[str] = set()
    i = 0
    n = len(lines)

    while i < n:
        m = _SUBPART_LINE_RE.match((lines[i] or "").strip())
        if not m:
            i += 1
            continue

        part_id = (m.group("part") or "").lower()
        j = i + 1
        found_solution = False

        while j < n:
            is_sol, _norm = _normalize_solution_line(lines[j] or "")
            if is_sol:
                found_solution = True
                break

            # Another subpart marker before any solution => ambiguous, abort this candidate.
            m2 = _SUBPART_LINE_RE.match((lines[j] or "").strip())
            if m2:
                break
            j += 1

        if found_solution:
            stmt_md = "\n".join(lines[i:j]).strip()
            if stmt_md and part_id not in seen:
                out.append((part_id, stmt_md))
                seen.add(part_id)
            i = j + 1
            continue

        i += 1

    return out


def _count_solution_markers(md: str) -> int:
    cnt = 0
    for ln in (md or "").splitlines():
        is_sol, _norm = _normalize_solution_line(ln or "")
        if is_sol:
            cnt += 1
    return cnt


def _split_interleaved_subpart_solution_segments(md: str) -> Tuple[List[Tuple[str, str, str]], str]:
    """
    Parse interleaved chains in markdown like:
      (b) <statement...>
      Solution.
      <proof...>
      (c) <statement...>
      Solution.
      <proof...>

    Returns:
      - list of (part_id, statement_md, proof_md)
      - remaining markdown that could not be parsed by this pattern
    """
    lines = (md or "").splitlines()
    n = len(lines)
    i = 0
    segs: List[Tuple[str, str, str]] = []
    consumed_to = 0
    seen_parts: set[str] = set()

    while i < n:
        while i < n and not (lines[i] or "").strip():
            i += 1
        if i >= n:
            break

        if _is_structural_boundary_md_line(lines[i] or ""):
            break

        m = _SUBPART_LINE_RE.match((lines[i] or "").strip())
        if not m:
            break

        part_id = (m.group("part") or "").lower()
        if part_id in seen_parts:
            # Repeated page artifacts are common; stop chain here to avoid duplicate emission.
            break
        j = i + 1
        sol_norm: List[str] = []
        found_sol = False
        while j < n:
            if _is_structural_boundary_md_line(lines[j] or ""):
                break
            is_sol, norm = _normalize_solution_line(lines[j] or "")
            if is_sol:
                found_sol = True
                sol_norm = norm
                break
            j += 1

        if not found_sol:
            break

        k = j + 1
        while k < n:
            if _is_structural_boundary_md_line(lines[k] or ""):
                break
            if _SUBPART_LINE_RE.match((lines[k] or "").strip()):
                break
            k += 1

        stmt_md = "\n".join(lines[i:j]).strip()
        proof_lines: List[str] = []
        if len(sol_norm) > 1:
            proof_lines.extend(sol_norm[1:])
        proof_lines.extend(lines[j + 1 : k])
        proof_md = "\n".join(proof_lines).strip()

        if stmt_md:
            segs.append((part_id, stmt_md, proof_md))
            seen_parts.add(part_id)
        consumed_to = k
        i = k

    if not segs:
        # Fallback for noisy OCR/LLM output:
        # sometimes later "Solution." markers are dropped, leaving:
        #   (b) <statement line>
        #   <proof body...>
        #   (c) <statement line>
        #   <proof body...>
        # In this mode, split by subpart markers only.
        if _count_solution_markers(md) == 0:
            lines2 = (md or "").splitlines()
            marks: List[Tuple[int, str]] = []
            for idx2, ln2 in enumerate(lines2):
                if _is_structural_boundary_md_line(ln2 or ""):
                    break
                m2 = _SUBPART_LINE_RE.match((ln2 or "").strip())
                if m2:
                    pid2 = (m2.group("part") or "").lower()
                    marks.append((idx2, pid2))
            if marks:
                out2: List[Tuple[str, str, str]] = []
                seen2: set[str] = set()
                for k2, (start_i, pid2) in enumerate(marks):
                    if pid2 in seen2:
                        continue
                    end_i = marks[k2 + 1][0] if (k2 + 1) < len(marks) else len(lines2)
                    seg_lines = lines2[start_i:end_i]
                    if not seg_lines:
                        continue
                    stmt_md2 = (seg_lines[0] or "").strip()
                    proof_md2 = "\n".join(seg_lines[1:]).strip()
                    if stmt_md2:
                        out2.append((pid2, stmt_md2, proof_md2))
                        seen2.add(pid2)
                if out2:
                    first_start = marks[0][0]
                    prefix_rest = "\n".join(lines2[:first_start]).strip()
                    return out2, prefix_rest
        return [], (md or "").strip()

    rest_md = "\n".join(lines[consumed_to:]).strip()
    return segs, rest_md


def _drop_leading_statement_before_solution(md: str) -> str:
    """
    For subpart chunks inside a proof, remove leading statement text before the first
    Solution marker, if present. Keeps proof bodies cleaner in OCR-interleaved cases.
    """
    lines = (md or "").splitlines()
    for i, ln in enumerate(lines):
        is_sol, norm = _normalize_solution_line(ln or "")
        if not is_sol:
            continue
        out: List[str] = []
        # Keep same-line solution tail, if any.
        if len(norm) > 1:
            out.extend(norm[1:])
        out.extend(lines[i + 1 :])
        return "\n".join(out).strip()
    return (md or "").strip()


def _split_proof_markdown_rule_based(md: str) -> Tuple[str, str]:
    """
    Deterministic split for proof chunks that begin with a Solution marker.

    Returns:
      - first solution body markdown (without the first literal "Solution.")
      - trailing markdown that starts from the first detected
        "(part statement) ... Solution." chain, if any

    This avoids relying solely on LLM splitting for interleaved multipart layouts
    like Exercise 3.1 / 2.17.
    """
    lines = (md or "").splitlines()
    if not lines:
        return "", ""

    # Locate first Solution marker.
    sol_i = -1
    sol_norm: List[str] = []
    for i, ln in enumerate(lines):
        is_sol, norm = _normalize_solution_line(ln or "")
        if is_sol:
            sol_i = i
            sol_norm = norm
            break
    if sol_i < 0:
        return "", (md or "").strip()

    body_lines: List[str] = []
    if len(sol_norm) > 1:
        body_lines.extend(sol_norm[1:])
    body_lines.extend(lines[sol_i + 1 :])

    if not body_lines:
        return "", ""

    marks: List[int] = []
    for i, ln in enumerate(body_lines):
        if _SUBPART_LINE_RE.match((ln or "").strip()):
            marks.append(i)

    rest_start = -1
    for k, mpos in enumerate(marks):
        next_pos = marks[k + 1] if (k + 1) < len(marks) else len(body_lines)
        has_solution = False
        for j in range(mpos + 1, next_pos):
            is_sol, _norm = _normalize_solution_line(body_lines[j] or "")
            if is_sol:
                has_solution = True
                break
        if has_solution:
            rest_start = mpos
            break

    if rest_start < 0:
        return "\n".join(body_lines).strip(), ""

    p_md = "\n".join(body_lines[:rest_start]).strip()
    r_md = "\n".join(body_lines[rest_start:]).strip()
    return p_md, r_md


def _normalize_subpart_line_starts(md: str, expected_parts: List[str]) -> str:
    """
    Lightweight repair: if a missing marker like '(b)' appears inline,
    move it to line start by inserting a newline before that token only.
    """
    text = md or ""
    if not text.strip() or not expected_parts:
        return text

    prelude, parts = split_subparts(text)
    found = {p for p, _ in parts}
    missing = [p for p in expected_parts if p not in found]
    # Conservative rule: if nothing is recognized yet, avoid forcing "(a)" out of prose.
    # We only try later parts first (b/c/...) in this case.
    if not found and len(missing) > 1:
        missing = missing[1:]
    if not missing:
        return text

    lines = text.splitlines()
    for miss in missing:
        # Only split when marker looks like a new item after sentence punctuation.
        tok_re = re.compile(rf"(?:^|[.;:]\s+)\(\s*{re.escape(miss)}\s*\)(?=\s+[A-Za-z])")
        done = False
        for i, ln in enumerate(lines):
            if _SUBPART_LINE_RE.match((ln or "").strip()):
                continue
            m = tok_re.search(ln or "")
            if not m:
                continue
            marker_m = re.search(rf"\(\s*{re.escape(miss)}\s*\)", (ln or "")[m.start():])
            if not marker_m:
                continue
            idx = m.start() + marker_m.start()
            if idx <= 0:
                continue
            left = ln[:idx].rstrip()
            right = ln[idx:].lstrip()
            if not left or not right:
                continue
            lines[i] = left
            lines.insert(i + 1, right)
            done = True
            break
        if done:
            # Re-evaluate progressively; one insertion may expose further markers.
            text = "\n".join(lines)
            lines = text.splitlines()
    return "\n".join(lines)


def _exercise_num_from_id(ex_id: str) -> str:
    m = _EXERCISE_NUM_RE.match((ex_id or "").strip())
    return (m.group("num") if m else "").strip()


def _build_block_open(type_name: str, label: str, *, exercise: str = "", part: str = "", refs: str = "") -> str:
    # Extend existing BLOCK sentinel with machine-friendly attrs for texTojson.
    attrs = [f"type={type_name}", f'label="{escape_attr(label)}"']
    if exercise:
        attrs.append(f'exercise="{escape_attr(exercise)}"')
    if part:
        attrs.append(f'part="{escape_attr(part)}"')
    if refs:
        attrs.append(f'refs="{escape_attr(refs)}"')
    return "%<BLOCK " + " ".join(attrs) + ">"


def _wrap_subpart_block(type_name: str, exercise_num: str, part_id: str, latex_body: str) -> str:
    label = f"Exercise {exercise_num}({part_id})" if exercise_num else f"({part_id})"
    body = (latex_body or "").strip()
    refs_attr = _theorem_refs_attr_from_text(body)
    open_line = _build_block_open(type_name, label, exercise=exercise_num, part=part_id, refs=refs_attr)
    return "\n".join([open_line, body, "%</BLOCK>"]).strip()


def _collect_subpart_pairs_for_warn(tex: str) -> Dict[str, Dict[str, set[str]]]:
    """
    Parse subpart sentinel attrs for quick coverage checks.
    """
    out: Dict[str, Dict[str, set[str]]] = {}
    # Parse only opening sentinels; this keeps the check independent of LaTeX layout.
    for m in re.finditer(r'(?m)^%<BLOCK\s+([^>]+)>\s*$', tex or ""):
        attrs_txt = m.group(1) or ""
        attrs = dict(re.findall(r'([a-zA-Z_]+)="([^"]*)"', attrs_txt))
        type_m = re.search(r"\btype=([a-zA-Z_]+)\b", attrs_txt)
        t = (type_m.group(1) if type_m else "").strip()
        ex = (attrs.get("exercise") or "").strip()
        part = (attrs.get("part") or "").strip().lower()
        if not ex or not part:
            continue
        if t not in {"subpart_statement", "subpart_proof"}:
            continue
        out.setdefault(ex, {"stmt": set(), "proof": set()})
        if t == "subpart_statement":
            out[ex]["stmt"].add(part)
        else:
            out[ex]["proof"].add(part)
    return out


def _extract_blocks_for_exercise(tex: str, exercise_num: str) -> Dict[str, Dict[str, str]]:
    """
    Extract statement/proof block body texts for a given exercise.
    Returns {"stmt": {part_id: body, ...}, "proof": {part_id: body, ...}}
    """
    result: Dict[str, Dict[str, str]] = {"stmt": {}, "proof": {}}
    block_re = re.compile(r'%<BLOCK\s+([^>]+)>\s*\n(.*?)%</BLOCK>', re.DOTALL)
    for m in block_re.finditer(tex or ""):
        attrs_txt = m.group(1) or ""
        body = m.group(2).strip()
        attrs = dict(re.findall(r'([a-zA-Z_]+)="([^"]*)"', attrs_txt))
        type_m = re.search(r"\btype=([a-zA-Z_]+)\b", attrs_txt)
        t = (type_m.group(1) if type_m else "").strip()
        ex = (attrs.get("exercise") or "").strip()
        part = (attrs.get("part") or "").strip().lower()
        if ex != exercise_num or not part:
            continue
        if t == "subpart_statement":
            result["stmt"][part] = body
        elif t == "subpart_proof":
            result["proof"][part] = body
    return result


def llm_verify_subpart_alignment(
    client: "OpenAI",
    model: str,
    exercise_num: str,
    stmt_parts: set,
    proof_parts: set,
    blocks: Dict[str, Dict[str, str]],
    max_tokens: int = 600,
) -> str:
    """
    Ask LLM to verify whether a subpart mismatch is a real error.
    Returns LLM diagnosis as a string (empty on failure).
    """
    miss_proof = sorted(stmt_parts - proof_parts)
    miss_stmt = sorted(proof_parts - stmt_parts)
    payload = {
        "exercise": exercise_num,
        "statement_parts": sorted(stmt_parts),
        "proof_parts": sorted(proof_parts),
        "missing_proof_for": miss_proof,
        "extra_proof_for": miss_stmt,
        "statement_content": {p: blocks["stmt"].get(p, "")[:400] for p in stmt_parts},
        "proof_content": {p: blocks["proof"].get(p, "")[:400] for p in proof_parts},
    }
    prompt = (
        "You are checking subpart alignment for a math exercise LaTeX extraction.\n"
        "Given statement parts and proof parts for an exercise, verify if the mismatch is real.\n"
        "Output ONLY one JSON object with keys:\n"
        "  'verdict': 'mismatch' | 'ok'\n"
        "  'explanation': str (brief reason)\n"
        "  'missing_proof': list of part ids truly missing proofs\n"
        "  'extra_proof': list of truly extra proof part ids\n\n"
        "Input:\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    try:
        raw = _chat_completion_text(
            client,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception:
        return ""
    return (raw or "").strip()


def warn_subpart_mismatch(
    tex: str,
    *,
    client: Optional["OpenAI"] = None,
    model: str = "",
    enable_llm_verify: bool = False,
    max_tokens: int = 600,
) -> None:
    pairs = _collect_subpart_pairs_for_warn(tex)
    ex_sorted = sorted(pairs.keys(), key=lambda x: [int(p) if p.isdigit() else p for p in x.split(".")])
    last_ex = ex_sorted[-1] if ex_sorted else ""
    for ex in ex_sorted:
        stmt = pairs[ex]["stmt"]
        proof = pairs[ex]["proof"]
        miss_proof = sorted(stmt - proof)
        miss_stmt = sorted(proof - stmt)
        has_mismatch = bool(miss_proof or miss_stmt)
        if miss_proof:
            suffix = " (possible tail truncation)" if ex == last_ex else ""
            print(f"[subpart-warn] Exercise {ex}: missing proof parts {miss_proof}{suffix}")
        if miss_stmt:
            print(f"[subpart-warn] Exercise {ex}: proof has extra parts {miss_stmt}")
        if has_mismatch and enable_llm_verify and client and model:
            blocks = _extract_blocks_for_exercise(tex, ex)
            llm_result = llm_verify_subpart_alignment(
                client, model, ex, stmt, proof, blocks, max_tokens=max_tokens
            )
            if llm_result:
                print(f"[subpart-llm-verify] Exercise {ex}: {llm_result}")


# =========================
# Main
# =========================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("in_md", type=str, help="Input Markdown (from pdfTomd_nn.py)")
    ap.add_argument("out_tex", type=str, help="Output LaTeX .tex")
    ap.add_argument("--workers", type=int, default=None, help="Override LLM worker count")
    ap.add_argument("--pdf", type=str, default=None, help="Optional PDF path for page-wise tag recovery")
    args = ap.parse_args()

    cfg = load_config()
    settings = load_settings()
    api_key = require_str(cfg, "api_key")
    base_url = get_cfg(cfg, "base_url", "https://aihubmix.com/v1")

    model = require_str(cfg, "model")
    max_tokens_think = int(get_setting(settings, "MDTOTEX_MAX_TOKENS", 2048))

    tag_model = model
    max_tokens_tag = int(get_setting(settings, "MDTOTEX_MAX_TOKENS_TAG", 1024))
    recover_interleaved_stmt = bool(get_setting(settings, "MDTOTEX_RECOVER_INTERLEAVED_SUBPART_STATEMENTS", True))
    strip_stmt_prefix_in_proof = bool(get_setting(settings, "MDTOTEX_STRIP_SUBPART_STATEMENT_PREFIX_IN_PROOF", True))
    enable_llm_resegment = bool(get_setting(settings, "MDTOTEX_ENABLE_LLM_RESEGMENT", True))
    enable_llm_subpart_verify = bool(get_setting(settings, "MDTOTEX_ENABLE_LLM_SUBPART_VERIFY", False))

    in_md = Path(args.in_md).expanduser().resolve()
    out_tex = Path(args.out_tex).expanduser().resolve()
    if not in_md.exists():
        print(f"ERROR: Markdown not found: {in_md}", file=sys.stderr)
        sys.exit(2)
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    workers: Optional[int] = args.workers
    if workers is None or workers <= 0:
        workers = int(get_setting(settings, "MDTOTEX_WORKERS", 4))

    md_text = in_md.read_text(encoding="utf-8")
    md_text = sanitize_ocr_markdown(md_text)

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=int(get_setting(settings, "MDTOTEX_TIMEOUT", 120)),
    )

    pages = split_markdown_pages(md_text)

    pdf_path = infer_pdf_path(in_md, args.pdf)
    if pdf_path is None:
        print("[tag] PDF not found -> skip page-wise tag recovery")
    else:
        print(f"[tag] PDF -> {pdf_path}")

    pages_fixed = pagewise_tag_recovery(
        pages=pages,
        pdf_path=pdf_path,
        client=client,
        tag_model=tag_model,
        max_tokens_tag=max_tokens_tag,
        workers=workers,
    )

    full_md = "\n\n".join([md for _p, md in sorted(pages_fixed, key=lambda x: x[0])]).strip()
    full_md = sanitize_ocr_markdown(full_md)

    full_md = attach_standalone_equation_numbers(full_md)

    anchored_md = inject_heading_sentinels(full_md)

    blocks = greedy_chunk_markdown(anchored_md)
    blocks = split_large_para_blocks(blocks, max_chars=int(get_setting(settings, "MDTOTEX_MAX_CHARS", 8000)))
    if not blocks:
        blocks = [Block(kind="para", env=None, md=full_md)]

    results: Dict[int, List[str]] = {}

    def _convert_one(idx: int, blk: Block) -> Tuple[int, List[str]]:
        if blk.kind == "heading":
            latex_h = heading_block_to_latex(blk.md)
            return idx, ([latex_h] if latex_h else [])

        if not blk.md.strip():
            return idx, []

        # Proof block: split markdown, convert body, code-wrap \begin{proof}...\end{proof}.
        if blk.kind == "proof":
            p_md, r_md = _split_proof_markdown_rule_based(blk.md)
            if not p_md and not r_md:
                p_md, r_md = markdown_proof_split_to_latex(
                    client=client,
                    model=model,
                    markdown=blk.md,
                    max_tokens=max_tokens_think,
                )
            if p_md:
                ex_num = ""
                expected_parts: List[str] = []
                for j in range(idx - 1, -1, -1):
                    b2 = blocks[j]
                    if b2.kind == "exercise":
                        ex_head, ex_body_md = _split_first_nonempty(b2.md)
                        ex_num = _exercise_num_from_id(ex_head)
                        _pre_stmt, stmt_parts = split_subparts(ex_body_md)
                        expected_parts = [pid for pid, _seg in stmt_parts]
                        if ex_num:
                            break

                # First-pass split; try local line-start normalization before LLM repair.
                p_md_norm = _normalize_subpart_line_starts(p_md, expected_parts)
                # Split proof body into optional prelude + (a)(b)(c) subparts.
                # Use expected-parts-bounded split so sub-steps inside the last part
                # (e.g., (a)(b)(c) inside (e)'s proof) are not mistaken as top-level splits.
                if expected_parts:
                    p_prelude_md, p_parts_md = split_subparts_expected(p_md_norm, expected_parts)
                else:
                    p_prelude_md, p_parts_md = split_subparts(p_md_norm)
                found_parts = [pid for pid, _seg in p_parts_md]
                if enable_llm_resegment and expected_parts and (set(expected_parts) - set(found_parts)):
                    p_md_checked = llm_resegment_subparts_markdown(
                        client=client,
                        model=model,
                        exercise_num=ex_num,
                        expected_parts=expected_parts,
                        proof_md=p_md,
                        max_tokens=max_tokens_think,
                    )
                    p_md_checked = _normalize_subpart_line_starts(p_md_checked, expected_parts)
                    p_prelude_md, p_parts_md = split_subparts_expected(p_md_checked, expected_parts)

                # Parse chained "(b)... Solution. ... (c)... Solution. ..." from trailing rest.
                extra_pairs, r_tail_md = _split_interleaved_subpart_solution_segments(r_md)

                proof_body_pieces: List[str] = []
                proof_by_part: Dict[str, str] = {}
                prelude_piece = ""
                found_part_ids = [pid for pid, _seg in p_parts_md]
                missing_expected = [pid for pid in expected_parts if pid not in set(found_part_ids)]
                if p_parts_md:
                    if p_prelude_md.strip():
                        # If proof prelude exists but expected part '(a)' is missing from parsed
                        # subparts, treat prelude as the proof for that first missing part.
                        if missing_expected:
                            pid0 = missing_expected[0]
                            prel_ltx = markdown_to_latex(client, model, p_prelude_md, max_tokens=max_tokens_think)
                            prel_ltx = strip_outer_env_wrapper(strip_outer_env_wrapper(prel_ltx, "proof"), "thm")
                            proof_by_part[pid0] = prel_ltx
                        else:
                            prel = markdown_to_latex(client, model, p_prelude_md, max_tokens=max_tokens_think)
                            prel = strip_outer_env_wrapper(strip_outer_env_wrapper(prel, "proof"), "thm")
                            if prel.strip():
                                prelude_piece = prel.strip()
                    for part_id, sub_md in p_parts_md:
                        sub_md_use = _drop_leading_statement_before_solution(sub_md) if strip_stmt_prefix_in_proof else sub_md
                        sub_ltx = markdown_to_latex(client, model, sub_md_use, max_tokens=max_tokens_think)
                        sub_ltx = strip_outer_env_wrapper(strip_outer_env_wrapper(sub_ltx, "proof"), "thm")
                        proof_by_part[part_id] = sub_ltx
                elif expected_parts:
                    # If this proof has no explicit (a)/(b)/... marker, map it to the first expected part.
                    # This makes multi-solution exercises still produce part-aligned proof blocks.
                    pid0 = expected_parts[0]
                    sub_ltx = markdown_to_latex(client, model, p_md, max_tokens=max_tokens_think)
                    sub_ltx = strip_outer_env_wrapper(strip_outer_env_wrapper(sub_ltx, "proof"), "thm")
                    proof_by_part[pid0] = sub_ltx
                elif p_prelude_md.strip():
                    prel = markdown_to_latex(client, model, p_prelude_md, max_tokens=max_tokens_think)
                    prel = strip_outer_env_wrapper(strip_outer_env_wrapper(prel, "proof"), "thm")
                    if prel.strip():
                        prelude_piece = prel.strip()

                # Shared-proof fallback: when a single proof body explicitly states equivalence
                # for another part (e.g., "Second part is similar."), replicate that proof to
                # missing statement parts to keep subpart alignment complete.
                if expected_parts and proof_by_part:
                    missing_after = [pid for pid in expected_parts if pid not in proof_by_part]
                    shared_hint = re.search(
                        r"(?i)\b(second|other|remaining)\s+part\s+is\s+similar\b|"
                        r"\b(same|similar)\s+(argument|proof)\b|"
                        r"\bsimilarly\b",
                        p_md or "",
                    )
                    if missing_after and shared_hint and len(proof_by_part) == 1:
                        src_part = next(iter(proof_by_part.keys()))
                        src_proof = proof_by_part[src_part]
                        for pid in missing_after:
                            proof_by_part[pid] = src_proof

                if prelude_piece:
                    proof_body_pieces.append(prelude_piece)
                if proof_by_part:
                    ordered_parts = expected_parts if expected_parts else sorted(proof_by_part.keys())
                    for pid in ordered_parts:
                        if pid in proof_by_part:
                            proof_body_pieces.append(
                                _wrap_subpart_block("subpart_proof", ex_num, pid, proof_by_part[pid])
                            )

                p_latex_body = "\n\n".join([x for x in proof_body_pieces if x.strip()]).strip()
                if not p_latex_body:
                    p_latex_body = markdown_to_latex(client, model, p_md, max_tokens=max_tokens_think)
                    p_latex_body = strip_outer_env_wrapper(strip_outer_env_wrapper(p_latex_body, "proof"), "thm")
                p_latex = _wrap_env("proof", p_latex_body, "")
                p_latex = insert_block_sentinels(p_latex)
                outs = [p_latex]

                # For interleaved multipart content, emit each pair as its own thm/proof.
                emitted_extra_parts: set[str] = set()
                for pid, stmt_md, proof_md in extra_pairs:
                    if pid in emitted_extra_parts:
                        continue
                    emitted_extra_parts.add(pid)
                    if stmt_md.strip():
                        stmt_ltx = markdown_to_latex(client, model, stmt_md, max_tokens=max_tokens_think)
                        stmt_ltx = strip_outer_env_wrapper(strip_outer_env_wrapper(stmt_ltx, "thm"), "proof")
                        stmt_blk = _wrap_subpart_block("subpart_statement", ex_num, pid, stmt_ltx)
                        stmt_thm = _wrap_env("thm", stmt_blk, "")
                        outs.append(insert_block_sentinels(stmt_thm))

                    if proof_md.strip():
                        proof_use = _drop_leading_statement_before_solution(proof_md) if strip_stmt_prefix_in_proof else proof_md
                        proof_ltx = markdown_to_latex(client, model, proof_use, max_tokens=max_tokens_think)
                        proof_ltx = strip_outer_env_wrapper(strip_outer_env_wrapper(proof_ltx, "proof"), "thm")
                        proof_blk = _wrap_subpart_block("subpart_proof", ex_num, pid, proof_ltx)
                        proof_env = _wrap_env("proof", proof_blk, "")
                        outs.append(insert_block_sentinels(proof_env))

                if r_tail_md.strip():
                    r_latex = markdown_to_latex(client, model, r_tail_md, max_tokens=max_tokens_think)
                    outs.append(insert_block_sentinels(r_latex))
                return idx, outs

            # fallback: convert whole block as-is
            _lead, body_md = _split_first_nonempty(blk.md)
            fallback_body = body_md if body_md else blk.md
            body_latex = markdown_to_latex(client, model, fallback_body, max_tokens=max_tokens_think)
            body_latex = strip_outer_env_wrapper(strip_outer_env_wrapper(body_latex, "proof"), "thm")
            latex = _wrap_env("proof", body_latex, "")
            latex = insert_block_sentinels(latex)
            return idx, [latex] if latex else []

        # Exercise block: code-wrap \begin{thm}...\end{thm} with deterministic title line.
        if blk.kind == "exercise":
            ex_id, ex_md_body = _split_first_nonempty(blk.md)
            ex_num = _exercise_num_from_id(ex_id)
            # Split statement body into optional prelude + (a)(b)(c) subparts.
            prelude_md, parts_md = split_subparts(ex_md_body)

            # OCR often interleaves later subpart statements inside the following proof block
            # as "(b) ... Solution." patterns across page breaks. Recover missing statement parts
            # so statement/proof subpart pairs stay aligned for downstream JSON extraction.
            if recover_interleaved_stmt and idx + 1 < len(blocks) and blocks[idx + 1].kind == "proof":
                # Always recover visible "(b)/(c)/..." statements from the following proof block.
                # This avoids dropping statement parts in layouts where proof markers are noisy.
                extra_stmt_parts = _extract_interleaved_subpart_statements_from_proof(blocks[idx + 1].md)
                if extra_stmt_parts:
                    existing = {pid for pid, _seg in parts_md}
                    for pid, stmt_md in extra_stmt_parts:
                        if pid not in existing:
                            parts_md.append((pid, stmt_md))
                            existing.add(pid)
                    parts_md = sorted(parts_md, key=lambda x: x[0])

            body_pieces: List[str] = []
            if prelude_md.strip():
                prel = markdown_to_latex(client, model, prelude_md, max_tokens=max_tokens_think)
                prel = strip_outer_env_wrapper(strip_outer_env_wrapper(prel, "thm"), "proof")
                if prel.strip():
                    body_pieces.append(prel.strip())

            for part_id, sub_md in parts_md:
                sub_ltx = markdown_to_latex(client, model, sub_md, max_tokens=max_tokens_think)
                sub_ltx = strip_outer_env_wrapper(strip_outer_env_wrapper(sub_ltx, "thm"), "proof")
                # Emit machine-readable subpart statement block.
                sub_blk = _wrap_subpart_block("subpart_statement", ex_num, part_id, sub_ltx)
                body_pieces.append(sub_blk)

            if body_pieces:
                body_latex = "\n\n".join(body_pieces).strip()
            else:
                body_latex = markdown_to_latex(client, model, ex_md_body, max_tokens=max_tokens_think) if ex_md_body else ""
                body_latex = strip_outer_env_wrapper(strip_outer_env_wrapper(body_latex, "thm"), "proof")
            latex = _wrap_env("thm", body_latex, ex_id)
            latex = insert_block_sentinels(latex)
            return idx, [latex] if latex else []

        # Other blocks: normal conversion
        latex = markdown_to_latex(client, model, blk.md, max_tokens=max_tokens_think)
        latex = insert_block_sentinels(latex)
        return idx, [latex] if latex else []

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = {ex.submit(_convert_one, i, blk): i for i, blk in enumerate(blocks)}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Semantic conversion (blocks)"):
            i, outs = fut.result()
            results[i] = outs

    # Preserve original order
    body_chunks: List[str] = []
    for i in range(len(blocks)):
        for piece in results.get(i, []):
            if piece and piece.strip():
                body_chunks.append(piece.strip())

    # -------- (5) Validation & assembly --------
    body_joined = "\n\n".join(body_chunks).strip()

    # (5b) Equation tag control:
    #      - If page-wise tag recovery ran, full_md contains the authoritative \tag{...} set.
    #      - Keep ONLY those tags, and de-duplicate accidental repeats in LaTeX.
    allowed_tags = set(re.findall(r"\\tag\{([^}]+)\}", full_md))
    body_joined = filter_and_dedupe_tags(body_joined, allowed_tags)

    # (5c) Heal math syntax (balance delimiters, fix illegal tag placement, fix begin/end pairing).
    body_joined = heal_latex_fragment(body_joined)

    # (5c1) Unwrap OCR mistakes: prose paragraphs accidentally wrapped in display math.
    body_joined = unwrap_prose_display_math(body_joined)

    # (5c2) Heal again after unwrapping (balances inline math delimiters, etc.).
    body_joined = heal_latex_fragment(body_joined)

    # (5c3) Final-pass equation-number normalization inside display math.
    #       Convert trailing textbook styles like "... \qquad (18.3)" -> "\tag{18.3}".
    body_joined = normalize_manual_eqnums_to_tags_in_latex(body_joined)

    # (5d) Prevent auto-numbering for untagged equation-like environments.
    #      (Untagged display math should be \[...\] or starred envs.)
    body_joined = star_all_equation_like_envs(body_joined)

    # (5e) Heal again after demotion (rare, but cheap and improves robustness).
    body_joined = heal_latex_fragment(body_joined)

    # (5f) Subpart coverage checks: statement(a,b,...) vs proof(a,b,...)
    warn_subpart_mismatch(
        body_joined,
        client=client if enable_llm_subpart_verify else None,
        model=model if enable_llm_subpart_verify else "",
        enable_llm_verify=enable_llm_subpart_verify,
        max_tokens=max_tokens_think,
    )

    tex = build_tex_document([body_joined])
    out_tex.write_text(tex, encoding="utf-8")
    print(f"DONE: {out_tex} (pages={len(pages)}, blocks={len(blocks)}, workers={workers})")


if __name__ == "__main__":
    main()
