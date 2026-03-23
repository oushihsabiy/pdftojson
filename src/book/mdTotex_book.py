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
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import BadRequestError, OpenAI

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None


# =========================
# LLM Prompts
# =========================

TAG_RECOVERY_PROMPT = (
    "You are given TWO inputs for the SAME PDF PAGE:\n"
    "A) OCR Markdown (may miss equation numbers)\n"
    "B) PDF plain text extracted from that page (has the correct equation numbers)\n"
    "\n"
    "Your ONLY task:\n"
    "- Recover missing equation numbers and insert LaTeX \\\\tag{...} into the corresponding DISPLAY equations.\n"
    "\n"
    "Hard rules (must follow):\n"
    "1) Do NOT rewrite, re-order, reflow, fix, or reformat any content.\n"
    "2) You may ONLY ADD '\\\\tag{...}' (and nothing else).\n"
    "3) Insert the tag INSIDE display math blocks (e.g. $$...$$ or \\\\[…\\\\] or equation/align envs).\n"
    "4) If a display equation already contains a number or already has \\\\tag, leave it unchanged.\n"
    "5) If you are not confident about a tag, DO NOTHING for that equation.\n"
    "\n"
    "Output ONLY the corrected Markdown (no fences, no commentary).\n"
    "\n"
    "<<<MD>>>\n"
)

LATEX_CONVERT_PROMPT = (
    "Convert the following OCR Markdown into LaTeX.\n"
    "Output ONLY LaTeX (no markdown fences, no commentary).\n"
    "\n"
    "The input MAY contain placeholder tokens like 'ZZZ_MATHBLOCK_0001_ZZZ'.\n"
    "These placeholders represent display-math blocks that will be restored later.\n"
    "You MUST keep every placeholder token EXACTLY unchanged, on its own line, and in the same order.\n"
    "Do NOT move placeholders, do NOT wrap them in any environment, and do NOT add any math delimiters around them.\n"
    "\n"
    "Strict block rules (IMPORTANT):\n"
    "A) Only wrap theorem-like environments if the input explicitly contains a numbered marker line:\n"
    "   - Theorem x.y(.z)  => \\\\begin{thm} ... \\\\end{thm}\n"
    "   - Lemma x.y(.z)    => \\\\begin{lem} ... \\\\end{lem}\n"
    "   - Proposition ...  => \\\\begin{prop} ... \\\\end{prop}\n"
    "   - Corollary ...    => \\\\begin{cor} ... \\\\end{cor}\n"
    "   - Definition ...   => \\\\begin{defn} ... \\\\end{defn}\n"
    "   - Algorithm ...    => \\begin{alg} ... \\end{alg}\n"
    "   If there is no explicit 'Definition N.N...' line in the Markdown, DO NOT invent a defn block.\n"
    "B) Only wrap a proof environment if the input explicitly contains a 'Proof.' marker.\n"
    "C) For EVERY theorem-like block, the FIRST LINE inside the environment MUST be the plain text title/number,\n"
    "   e.g. 'Theorem 2.1.' (do NOT use \\\\textbf for that line).\n"
    "\n"
    "Math rules:\n"
    "- Inline math: use $...$.\n"
    "- Display math: use \\\\[ ... \\\\] (do NOT use $$...$$).\n"
    "- IMPORTANT: If the input display math already contains \\\\tag{...}, output it using an amsmath display\n"
    "  environment that legally supports \\\\tag, e.g. \\\\begin{equation} ... \\\\end{equation} (or align).\n"
    "- CRITICAL: Do NOT invent any new \\tag{...}. Only keep tags that already appear in the input Markdown.\n"
    "\n"
    "Formatting:\n"
    "- Preserve paragraph structure.\n"
    "- Convert Markdown **bold** / *italic* to LaTeX \\\\textbf{} / \\\\emph{} where appropriate.\n"
    "- Preserve wording; do not rewrite content.\n"
    "\n"
    "Markdown:\n"
)

# Proof split prompt (greedy proof blocks may include trailing non-proof text)
PROOF_SPLIT_PROMPT = (
    "You are given OCR Markdown that CONTAINS an explicit 'Proof.' marker.\n"
    "The provided chunk may include extra non-proof paragraphs AFTER the proof ends.\n"
    "\n"
    "The input MAY contain placeholder tokens like 'ZZZ_MATHBLOCK_0001_ZZZ'.\n"
    "These placeholders represent display-math blocks that will be restored later.\n"
    "You MUST keep every placeholder token EXACTLY unchanged, on its own line, and in the same order.\n"
    "Do NOT move placeholders, do NOT wrap them in any environment, and do NOT add any math delimiters around them.\n"
    "\n"
    "Task:\n"
    "1) Convert to LaTeX.\n"
    "2) Decide where the proof logically ends (QED/∎/end-of-proof cue).\n"
    "3) Output TWO LaTeX parts in STRICT FORMAT using the separators exactly:\n"
    "<<<PROOF>>>\n"
    "<LaTeX proof environment ONLY: must include \\\\begin{proof} ... \\\\end{proof}>\n"
    "<<<REST>>>\n"
    "<Remaining LaTeX AFTER the proof ends (not inside proof). If nothing remains, output empty after this line.>\n"
    "\n"
    "Rules:\n"
    "- Do NOT output markdown fences or commentary.\n"
    "- Do NOT invent theorem/definition environments unless the input explicitly contains numbered marker lines.\n"
    "- Inline math: $...$ ; Display math: \\\\[ ... \\\\] (do NOT use $$...$$).\n"
    "- CRITICAL: Do NOT invent any new \\tag{...}. Only keep tags that already appear in the input Markdown.\n"
    "- Preserve paragraph structure and wording; do not rewrite content.\n"
    "- IMPORTANT: If the proof is already clearly ended inside the chunk, keep ONLY the proof text in <<<PROOF>>>.\n"
    "\n"
    "Markdown:\n"
)

ENVS = ["defn", "thm", "lem", "prop", "cor", "alg"]
ORIGIN_ENV = "origintext"


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


def _wrap_text_as_chat_response(text: str):
    class _Msg:
        def __init__(self, content: str):
            self.content = content

    class _Choice:
        def __init__(self, content: str):
            self.message = _Msg(content)

    class _Resp:
        def __init__(self, content: str):
            self.choices = [_Choice(content)]

    return _Resp(text)


def _chat_create_with_token_compat(client: OpenAI, **kwargs):
    """
    Prefer `max_completion_tokens`; fall back to `max_tokens` for older models.
    Also handles gateways that require stream=true.
    """

    def _call_with_stream_fallback(req: Dict[str, Any]):
        try:
            return client.chat.completions.create(**req)
        except Exception:
            # Some gateways may return non-JSON/empty body for non-stream chat calls.
            # Fallback to stream path whenever non-stream fails.
            stream_obj = client.chat.completions.create(stream=True, **req)
            return _wrap_text_as_chat_response(_collect_stream_text(stream_obj) or "")

    req = dict(kwargs)
    if "max_tokens" in req:
        req["max_completion_tokens"] = req.pop("max_tokens")
    try:
        return _call_with_stream_fallback(req)
    except BadRequestError as e:
        msg = str(e)
        if (
            "max_completion_tokens" in req
            and "unsupported_parameter" in msg
            and "max_completion_tokens" in msg
        ):
            fallback_req = dict(kwargs)
            return _call_with_stream_fallback(fallback_req)
        raise


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
    resp = _chat_create_with_token_compat(client, 
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    return (resp.choices[0].message.content or "").strip()


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
NUMSEC_LINE_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+(.+?)\s*$")


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

        # (B) Numeric section headings like '6.1 Delayed column generation'
        nm = NUMSEC_LINE_RE.match(ln.strip())
        if nm:
            out.append(HEADING_START)
            out.append(ln.strip())
            out.append(HEADING_END)
            i += 1
            continue

        # (C) Markdown headings
        hm = MD_HEADING_RE.match(ln.strip())
        if hm:
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

_STMT_START_RE = re.compile(
    r"^\s*(?:[*_`> ]*)"
    r"(Theorem|Lemma|Proposition|Corollary|Definition|Algorithm|Alg\.?)s?\s+"
    r"([0-9]+(?:\.[0-9]+)*)"
    r"\s*\.?\s*(?:[*_` ]*)"
    r"(.*)$",
    re.IGNORECASE,
)

_PROOF_START_RE = re.compile(
    r"^\s*(?:[*_`> ]*)Proof\s*[:.]?\s*(?:[*_` ]*)"
    r"(.*)$",
    re.IGNORECASE,
)


def _normalize_stmt_line(line: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    m = _STMT_START_RE.match(line.strip())
    if not m:
        return None, None, []
    kind_raw = (m.group(1) or "").strip()
    kind = "Algorithm" if kind_raw.lower().startswith("alg") else kind_raw.title()
    num = m.group(2)
    tail = (m.group(3) or "").strip()

    env_map = {
        "Theorem": "thm",
        "Lemma": "lem",
        "Proposition": "prop",
        "Corollary": "cor",
        "Definition": "defn",
        "Algorithm": "alg",
    }
    env = env_map.get(kind, "thm")

    stmt_id = f"{kind} {num}"
    first = f"{stmt_id}."
    lines = [first]
    if tail:
        tail = tail.lstrip(":-–—. ").rstrip()
        if tail:
            lines.append(tail)
    return env, stmt_id, lines


def _normalize_proof_line(line: str) -> Tuple[bool, List[str]]:
    # Proof block detection disabled: keep "Proof." as regular paragraph text.
    return False, []


@dataclass
class Block:
    kind: str              # "heading" | "stmt" | "proof" | "para"
    env: Optional[str]     # for stmt/proof: thm/lem/.../proof
    md: str                # markdown payload


def greedy_chunk_markdown(anchored_md: str) -> List[Block]:
    """
    Streaming greedy scan.
    Priority:
      1) Heading sentinel blocks: forced break
      2) New environment starts (stmt/proof): forced break
      3) Proof blocks are greedy: they include everything until next heading or env-start
         (we do NOT try to detect end-of-proof here)
    """
    lines = (anchored_md or "").splitlines()

    blocks: List[Block] = []

    cur_kind: Optional[str] = None
    cur_env: Optional[str] = None
    cur_stmt_id: Optional[str] = None
    cur_lines: List[str] = []

    def flush() -> None:
        nonlocal cur_kind, cur_env, cur_stmt_id, cur_lines
        if cur_kind is None:
            return
        text = "\n".join(cur_lines).strip()
        if text:
            blocks.append(Block(kind=cur_kind, env=cur_env, md=text))
        cur_kind = None
        cur_env = None
        cur_stmt_id = None
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

        # (B) stmt start
        env, stmt_id, norm_stmt_lines = _normalize_stmt_line(ln)
        if env is not None:
            if cur_kind == "stmt" and cur_stmt_id and stmt_id == cur_stmt_id:
                # repeated title line across pages -> treat as continuation
                if len(norm_stmt_lines) > 1:
                    cur_lines.extend(norm_stmt_lines[1:])
                i += 1
                continue

            flush()
            cur_kind = "stmt"
            cur_env = env
            cur_stmt_id = stmt_id
            cur_lines = list(norm_stmt_lines)
            i += 1
            continue

        # (C) proof boundary (without creating a proof block)
        # Keep theorem-ending behavior: a leading "Proof." should end current stmt.
        if _PROOF_START_RE.match(ln.strip()):
            if cur_kind == "stmt":
                flush()
            if cur_kind is None:
                cur_kind = "para"
                cur_env = None
                cur_stmt_id = None
                cur_lines = [ln]
            else:
                cur_lines.append(ln)
            i += 1
            continue

        # (D) normal line
        if cur_kind is None:
            cur_kind = "para"
            cur_env = None
            cur_stmt_id = None
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


# ---- NEW: wrap/balance Example blocks ----
_EXAMPLE_BOLD_RE = re.compile(
    r"^\s*\\textbf\{(?P<title>Example\s+\d+(?:\.\d+)*(?:\s*\([^}]*\))?)\}\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
_EXAMPLE_PLAIN_RE = re.compile(
    r"^\s*(?P<title>Example\s+\d+(?:\.\d+)*(?:\s*\([^)]*\))?)\s*(?P<rest>.*)$",
    re.IGNORECASE,
)

# Environments that should terminate an Example block (math envs are NOT included on purpose).
_EXAMPLE_END_ENV_RE = re.compile(r"^\s*\\begin\{(?:thm|lem|prop|cor|defn|alg)\}\s*$")


def _normalize_example_title(title: str) -> str:
    t = (title or "").strip()
    # Standardize punctuation: "Example 6.4" -> "Example 6.4."
    if t.endswith(":"):
        t = t[:-1].rstrip()
    if not t.endswith("."):
        t = t + "."
    return t


def wrap_and_balance_examples(latex: str) -> str:
    """
    Convert OCR/LLM output like:
        \\textbf{Example 6.4} ...
    or:
        Example 6.2 ...
    into a proper example environment, and ensure \\begin{example}/\\end{example} are balanced.

    Strategy (greedy but safe):
    - An Example block starts at a line that *begins* with an Example marker.
    - It ends when we hit:
        * a heading/page hard boundary (\\chapter/\\section/\\clearpage/% ===== Page ...)
        * a new Example marker
        * a theorem/proof environment begin (thm/lem/prop/cor/defn/alg)
    """
    lines = (latex or "").splitlines()
    out: List[str] = []

    env_stack: List[str] = []

    def in_env(env: str) -> bool:
        return any(e == env for e in env_stack)

    def top_level_inside_example() -> bool:
        # inside example and not inside any nested env beyond it
        return env_stack == ["example"]

    def push_env(line: str) -> None:
        mb = re.match(r"^\s*\\begin\{([^}]+)\}", line)
        if mb:
            env_stack.append(mb.group(1).strip())

    def pop_env(line: str) -> None:
        me = re.match(r"^\s*\\end\{([^}]+)\}", line)
        if not me:
            return
        env = me.group(1).strip()
        for k in range(len(env_stack) - 1, -1, -1):
            if env_stack[k] == env:
                del env_stack[k:]
                return

    def close_open_example() -> None:
        nonlocal env_stack
        if env_stack and env_stack[-1] == "example":
            out.append(r"\end{example}")
            env_stack.pop()

    def is_example_start(line: str) -> Tuple[bool, str, str]:
        """
        Returns (is_start, title, rest)
        """
        m = _EXAMPLE_BOLD_RE.match(line)
        if m:
            return True, (m.group("title") or ""), (m.group("rest") or "")
        m2 = _EXAMPLE_PLAIN_RE.match(line)
        if m2:
            return True, (m2.group("title") or ""), (m2.group("rest") or "")
        return False, "", ""

    i = 0
    while i < len(lines):
        line = lines[i]

        # If we're inside an example (top-level) but encounter a hard boundary, close before it.
        if top_level_inside_example():
            if _is_hard_boundary(line) or _EXAMPLE_END_ENV_RE.match(line):
                close_open_example()

        # Handle explicit begin/end of example (and balance)
        if re.match(r"^\s*\\begin\{example\}\s*$", line):
            out.append(r"\begin{example}")
            env_stack.append("example")
            i += 1
            continue

        if re.match(r"^\s*\\end\{example\}\s*$", line):
            if in_env("example"):
                out.append(r"\end{example}")
                # pop last matching example
                for k in range(len(env_stack) - 1, -1, -1):
                    if env_stack[k] == "example":
                        del env_stack[k:]
                        break
            # else: drop unmatched end
            i += 1
            continue

        # If a new example starts, close previous (if we were in one) and open a new one.
        if not env_stack:
            is_start, title, rest = is_example_start(line)
            if is_start:
                title_line = _normalize_example_title(title)
                rest_line = (rest or "").strip()
                rest_line = rest_line.lstrip(":-–—. ").rstrip()

                out.append(r"\begin{example}")
                out.append(title_line)
                env_stack.append("example")

                if rest_line:
                    out.append(rest_line)

                i += 1
                continue

        # Inside an example (top-level): if a new Example marker appears, close and restart.
        if top_level_inside_example():
            is_start2, title2, rest2 = is_example_start(line)
            if is_start2:
                close_open_example()
                title_line = _normalize_example_title(title2)
                rest_line = (rest2 or "").strip()
                rest_line = rest_line.lstrip(":-–—. ").rstrip()

                out.append(r"\begin{example}")
                out.append(title_line)
                env_stack.append("example")
                if rest_line:
                    out.append(rest_line)
                i += 1
                continue

        # passthrough line
        out.append(line)
        push_env(line)
        pop_env(line)
        i += 1

    # Close any remaining open example at end of doc
    if top_level_inside_example():
        close_open_example()

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

    # Extract any \tag tokens and move them OUTSIDE an eventual aligned wrapper
    tags = [t.strip() for t in _TAG_TOKEN_RE.findall(body) if t.strip()]
    body_wo_tags = _TAG_TOKEN_RE.sub("", body).strip()

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

_STANDALONE_EQNUM_LINE_RE = re.compile(r"^\s*\((\d+(?:\.\d+)*)\)\s*$")

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

@retry(
    reraise=True,
    stop=stop_after_attempt(2),
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
    md_in = (markdown or "").strip()

    md_ph, mapping, seq = replace_display_math_with_placeholders(md_in)
    prompt = LATEX_CONVERT_PROMPT + md_ph.strip()

    resp = _chat_create_with_token_compat(client, 
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    raw = resp.choices[0].message.content or ""
    raw = strip_code_fences(raw)
    raw = strip_outer_document(raw)

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
    m = re.search(r"(?s)<<<PROOF>>>\s*(.*?)\s*<<<REST>>>\s*(.*)\s*$", t)
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
    stop=stop_after_attempt(2),
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
    Ask the LLM to locate the logical end of a proof (QED) and split the chunk into:
      - proof part (to be wrapped in the proof environment)
      - the following non-proof text

    We preserve display-math blocks via placeholders to prevent \\tag{...} from moving across prose.
    """
    md_in = (markdown or "").strip()
    md_ph, mapping, seq = replace_display_math_with_placeholders(md_in)

    prompt = PROOF_SPLIT_PROMPT + md_ph.strip()
    resp = _chat_create_with_token_compat(client, 
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    raw = resp.choices[0].message.content or ""
    raw = strip_code_fences(raw)
    raw = strip_outer_document(raw)

    proof_part, rest_part = _split_proof_output(raw)

    if mapping:
        found = _PLACEHOLDER_RE.findall(proof_part + "\n" + rest_part)
        # If the model tampers, we still restore what we can (best-effort).
        proof_part = restore_display_math_placeholders(proof_part, mapping)
        rest_part = restore_display_math_placeholders(rest_part, mapping)

    proof_tex = heal_latex_fragment(proof_part)
    rest_tex = heal_latex_fragment(rest_part)
    return proof_tex.strip(), rest_tex.strip()

# =========================
# Block sentinels (for texTojson_new)
# =========================

ENV_BLOCK_RE = re.compile(
    r"\\begin\{(?P<env>origintext|defn|thm|lem|prop|cor|alg)\}\s*(?P<body>.*?)\\end\{\1\}",
    re.DOTALL,
)

NUMBERED_TITLE_RE = re.compile(
    r"\b(Theorem|Lemma|Proposition|Corollary|Definition|Algorithm|Alg\.?)\s+([0-9]+(?:\.[0-9]+)*)\s*\.?",
    re.IGNORECASE,
)


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

    m = NUMBERED_TITLE_RE.search(head_plain)
    if m:
        kind_raw = (m.group(1) or "").strip()
        kind = "Algorithm" if kind_raw.lower().startswith("alg") else kind_raw.title()
        num = m.group(2)
        return f"{kind} {num}"

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


def normalize_to_origintext_envs(latex: str) -> str:
    s = latex or ""
    # Normalize theorem/proof-like envs into one uniform environment.
    s = re.sub(
        r"\\begin\{(?:defn|thm|lem|prop|cor|alg)\}(?:\[[^\]]*\])?",
        rf"\\begin{{{ORIGIN_ENV}}}",
        s,
    )
    s = re.sub(
        r"\\end\{(?:defn|thm|lem|prop|cor|alg)\}",
        rf"\\end{{{ORIGIN_ENV}}}",
        s,
    )
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
        out.append(f'%<BLOCK type={ORIGIN_ENV} label="{escape_attr(label)}">\n')
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
    Convert a heading block (content between HEADING_START/END) into LaTeX heading commands.

    Notes:
    - If the heading begins with a numeric prefix (e.g. '6.3 ...'), we map by numeric depth, NOT by
      markdown '#' level (OCR often mis-uses heading levels).
    """
    lines = [ln.strip() for ln in (heading_md or "").splitlines() if ln.strip()]
    if not lines:
        return ""

    # SECTION/CHAPTER style
    m = SEC_LINE_RE.match(lines[0])
    if m:
        kind = m.group(1).title()  # Section / Chapter
        num = m.group(2)
        title = lines[1] if len(lines) > 1 else ""
        if title:
            text = f"{kind} {num}: {title}"
        else:
            text = f"{kind} {num}"

        if kind.lower() == "chapter":
            return rf"\chapter{{{text}}}".strip()
        return rf"\section{{{text}}}".strip()

    # Extract heading text and (optional) markdown level
    level: int | None = None
    first = lines[0]
    hm = MD_HEADING_RE.match(first)
    if hm:
        level = len(hm.group(1))
        head_text = hm.group(2).strip()
    else:
        head_text = first.strip()

    # Numeric section heading style (e.g. '6.1 Delayed column generation')
    nm = NUMSEC_LINE_RE.match(head_text)
    if nm:
        num = nm.group(1)
        title = nm.group(2).strip()
        depth = num.count(".")
        if depth == 1:
            cmd = "section"
        elif depth == 2:
            cmd = "subsection"
        elif depth == 3:
            cmd = "subsubsection"
        else:
            cmd = "paragraph"
        return rf"\{cmd}{{{num} {title}}}".strip()

    # Markdown heading style (non-numeric)
    if level is not None:
        text = head_text
        if level == 1:
            return rf"\section{{{text}}}"
        if level == 2:
            return rf"\subsection{{{text}}}"
        if level == 3:
            return rf"\subsubsection{{{text}}}"
        return rf"\paragraph{{{text}}}"

    # fallback: treat as \section
    if len(lines) == 1:
        return rf"\section{{{lines[0]}}}"
    return rf"\section{{{lines[0]}: {lines[1]}}}"




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
\geometry{margin=1in}

% OCR sometimes outputs non-standard optimization operators
\providecommand{\minimize}{\min}
\providecommand{\maximize}{\max}

% Unified origin environment for all theorem/proof-like content
\newenvironment{origintext}{}{}
\newtheorem*{example}{Example}

\begin{document}
"""
    end = r"""
\end{document}
"""
    return preamble + "\n\n".join([c for c in body_chunks if c and c.strip()]).rstrip() + "\n" + end


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

    full_md = attach_standalone_equation_numbers(full_md)

    anchored_md = inject_heading_sentinels(full_md)

    blocks = greedy_chunk_markdown(anchored_md)
    blocks = split_large_para_blocks(blocks, max_chars=int(get_setting(settings, "MDTOTEX_MAX_CHARS", 8000)))
    if not blocks:
        blocks = [Block(kind="para", env=None, md=full_md)]

    results: Dict[int, List[str]] = {}

    block_timeout_sec = max(30, int(get_setting(settings, "MDTOTEX_BLOCK_TIMEOUT_SEC", 180)))

    def _fallback_convert_without_llm(idx: int, blk: Block, reason: str = "timeout") -> List[str]:
        # Conservative fallback to avoid pipeline hangs on a single LLM call.
        if blk.kind == "heading":
            latex_h = heading_block_to_latex(blk.md)
            return [latex_h] if latex_h else []
        body = (blk.md or "").strip()
        if not body:
            return []
        wrapped = "\\begin{origintext}\n" + body + "\n\\end{origintext}"
        wrapped = normalize_to_origintext_envs(wrapped)
        wrapped = insert_block_sentinels(wrapped)
        print(f"[warn] semantic block fallback idx={idx} kind={blk.kind} reason={reason}")
        return [wrapped] if wrapped else []

    def _convert_one(idx: int, blk: Block) -> Tuple[int, List[str]]:
        if blk.kind == "heading":
            latex_h = heading_block_to_latex(blk.md)
            return idx, ([latex_h] if latex_h else [])

        if not blk.md.strip():
            return idx, []

        # Proof: ask model to split proof/rest
        if blk.kind == "proof":
            p_latex, r_latex = markdown_proof_split_to_latex(
                client=client,
                model=model,
                markdown=blk.md,
                max_tokens=max_tokens_think,
            )
            if p_latex:
                p_latex = normalize_to_origintext_envs(p_latex)
                p_latex = insert_block_sentinels(p_latex)
                outs = [p_latex]
                if r_latex.strip():
                    r_latex = normalize_to_origintext_envs(r_latex)
                    outs.append(insert_block_sentinels(r_latex))
                return idx, outs

            # fallback: convert whole block as-is
            latex = markdown_to_latex(client, model, blk.md, max_tokens=max_tokens_think)
            latex = normalize_to_origintext_envs(latex)
            latex = insert_block_sentinels(latex)
            return idx, [latex] if latex else []

        # Other blocks: normal conversion
        latex = markdown_to_latex(client, model, blk.md, max_tokens=max_tokens_think)
        latex = normalize_to_origintext_envs(latex)
        latex = insert_block_sentinels(latex)
        return idx, [latex] if latex else []

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = {ex.submit(_convert_one, i, blk): (i, blk) for i, blk in enumerate(blocks)}
        pending = set(futs.keys())
        with tqdm(total=len(futs), desc="Semantic conversion (blocks)") as pbar:
            while pending:
                done, pending = wait(pending, timeout=block_timeout_sec, return_when=FIRST_COMPLETED)
                if done:
                    for fut in done:
                        i, blk = futs[fut]
                        try:
                            ii, outs = fut.result()
                            results[ii] = outs
                        except Exception as e:
                            results[i] = _fallback_convert_without_llm(i, blk, reason=f"exception:{type(e).__name__}")
                    pbar.update(len(done))
                    continue

                # No task completed within timeout window: fallback remaining tasks
                # to guarantee forward progress for long-tail/blocked calls.
                for fut in list(pending):
                    i, blk = futs[fut]
                    if i not in results:
                        results[i] = _fallback_convert_without_llm(i, blk, reason=f"no_progress_{block_timeout_sec}s")
                    fut.cancel()
                    pending.remove(fut)
                    pbar.update(1)

    # Preserve original order
    body_chunks: List[str] = []
    for i in range(len(blocks)):
        for piece in results.get(i, []):
            if piece and piece.strip():
                body_chunks.append(piece.strip())

    # -------- (5) Validation & assembly --------
    body_joined = "\n\n".join(body_chunks).strip()

    # (5a0) Wrap standalone figure captions into figure environments.
    body_joined = wrap_figure_captions(body_joined)

    # (5a) Figure placeholders: remove \includegraphics (no image files) but keep the caption.
    body_joined = fix_missing_figures(body_joined)

    # (5a1) Wrap/balance Example blocks.
    body_joined = wrap_and_balance_examples(body_joined)

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

    # (5d) Prevent auto-numbering for untagged equation-like environments.
    #      (Untagged display math should be \[...\] or starred envs.)
    body_joined = star_all_equation_like_envs(body_joined)

    # (5e) Heal again after demotion (rare, but cheap and improves robustness).
    body_joined = heal_latex_fragment(body_joined)

    tex = build_tex_document([body_joined])
    out_tex.write_text(tex, encoding="utf-8")
    print(f"DONE: {out_tex} (pages={len(pages)}, blocks={len(blocks)}, workers={workers})")


if __name__ == "__main__":
    main()
