#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Paper-oriented Markdown to LaTeX conversion."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI


# ---------------- Prompt ----------------

LATEX_CONVERT_PROMPT = (
    "Convert the following OCR/converted Markdown into LaTeX.\n"
    "Output ONLY LaTeX (no markdown fences, no commentary).\n"
    "\n"
    "IMPORTANT conversion constraints:\n"
    "1) Do NOT use \\maketitle.\n"
    "2) Preserve bracket references like [11], [7] as-is (do NOT convert to \\cite).\n"
    "3) Keep existing LaTeX math commands unchanged when they already appear.\n"
    "\n"
    "Strict block rules (IMPORTANT):\n"
    "A) Only wrap theorem-like environments if the input explicitly contains a numbered marker line on its own line:\n"
    "   - Theorem x.y(.z)  => \\begin{thm} ... \\end{thm}\n"
    "   - Lemma x.y(.z)    => \\begin{lem} ... \\end{lem}\n"
    "   - Proposition ...  => \\begin{prop} ... \\end{prop}\n"
    "   - Corollary ...    => \\begin{cor} ... \\end{cor}\n"
    "   - Definition ...   => \\begin{defn} ... \\end{defn}\n"
    "   Marker line example (must be standalone):\n"
    "     Theorem 2.1.\n"
    "B) Only wrap a proof environment if the input explicitly contains a standalone 'Proof.' marker line.\n"
    "C) For EVERY theorem-like block, the FIRST LINE inside the environment MUST be the plain text title/number,\n"
    "   e.g. 'Theorem 2.1.' (do NOT use \\textbf for that line).\n"
    "\n"
    "Math rules:\n"
    "- Inline math: use $...$.\n"
    "- Display math: use \\[ ... \\] (do NOT use $$...$$).\n"
    "\n"
    "Formatting:\n"
    "- Preserve paragraph structure.\n"
    "- Convert Markdown **bold** / *italic* to LaTeX \\textbf{} / \\emph{} where appropriate.\n"
    "\n"
    "Markdown:\n"
)


ENVS = ["defn", "thm", "lem", "prop", "cor", "proof"]


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


# ---------------- markdown splitting ----------------

def split_markdown_pages(md_text: str) -> List[Tuple[int, str]]:
    parts = re.split(r"(?m)^\s*<!--\s*PAGE\s+(\d+)\s*-->\s*$", md_text)
    out: List[Tuple[int, str]] = []
    i = 1
    while i + 1 < len(parts):
        page_num = int(parts[i])
        content = parts[i + 1].strip()
        out.append((page_num, content))
        i += 2
    if not out:
        out = [(1, md_text.strip())]
    return out


# ---------------- heading extraction (deterministic) ----------------

MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SEC_LINE_RE = re.compile(r"^\s*(SECTION|CHAPTER)\s+(\d+)\s*$", re.IGNORECASE)

# Paper-style numbered headings:
#  "1. Introduction"
#  "4.1. Minimax strategies..."
#  "4.1.1. Something..."
NUM_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.\s+(.+?)\s*$")

def numbered_heading_to_latex(num: str, title: str) -> str:
    parts = num.split(".")
    level = len(parts)  # 1 -> section, 2 -> subsection, 3 -> subsubsection, 4+ -> paragraph
    full = f"{num}. {title}".strip()
    if level <= 1:
        return rf"\section{{{full}}}"
    if level == 2:
        return rf"\subsection{{{full}}}"
    if level == 3:
        return rf"\subsubsection{{{full}}}"
    return rf"\paragraph{{{full}}}"

def extract_and_convert_headings(md: str) -> Tuple[List[str], str]:
    """
    Returns (latex_heading_lines, remaining_markdown).

    Strategy:
    1) If page starts with 'SECTION N' or 'CHAPTER N', take next non-empty line as title:
       \\section{Section N: Title}
    2) Convert explicit Markdown headings (#/##/###...) deterministically
    3) Convert paper numbered headings like '4.1. Title' anywhere
    4) NO heuristic from first lines (disabled for papers)
    """
    lines = md.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)

    latex_lines: List[str] = []

    # (A) SECTION/CHAPTER at page top
    if lines:
        m = SEC_LINE_RE.match(lines[0].strip())
        if m:
            kind = m.group(1).title()
            num = m.group(2)
            j = 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            title = lines[j].strip() if j < len(lines) else ""
            if title:
                latex_lines.append(rf"\section{{{kind} {num}: {title}}}")
                del lines[: j + 1]
                while lines and not lines[0].strip():
                    lines.pop(0)

    # (B)(C) Explicit MD headings + numbered headings anywhere
    remaining: List[str] = []
    for line in lines:
        s = line.strip()
        hm = MD_HEADING_RE.match(s)
        if hm:
            level = len(hm.group(1))
            text = hm.group(2).strip()
            if level == 1:
                latex_lines.append(rf"\section{{{text}}}")
            elif level == 2:
                latex_lines.append(rf"\subsection{{{text}}}")
            elif level == 3:
                latex_lines.append(rf"\subsubsection{{{text}}}")
            else:
                latex_lines.append(rf"\paragraph{{{text}}}")
            continue

        nm = NUM_HEADING_RE.match(s)
        if nm:
            num = nm.group(1).strip()
            title = nm.group(2).strip()
            latex_lines.append(numbered_heading_to_latex(num, title))
            continue

        remaining.append(line)

    return latex_lines, "\n".join(remaining).strip()


# ---------------- Markdown pre-normalization for blocks ----------------

# Accept bold or plain theorem markers; ensure they become standalone marker lines.
THEOREM_MARK_RE = re.compile(
    r"""^\s*(?:\*\*)?
        (?P<kind>Theorem|Lemma|Proposition|Corollary|Definition)
        \s+(?P<num>\d+(?:\.\d+)*)
        \s*\.?\s*
        (?:\*\*)?
        (?P<rest>\s+.+)?\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

PROOF_MARK_RE = re.compile(r"^\s*(?:\*\*)?\s*Proof\s*\.?\s*(?:\*\*)?\s*(?P<rest>.*)\s*$", re.IGNORECASE)

def normalize_markdown_block_markers(md: str) -> str:
    """
    Turn various marker styles into:
      Theorem 1.
      <statement continues...>

      Proof.
      <proof continues...>

    Key: marker must be its own line to satisfy the LLM rule.
    """
    out_lines: List[str] = []
    for line in md.splitlines():
        s = line.strip()
        if not s:
            out_lines.append(line)
            continue

        pm = PROOF_MARK_RE.match(line)
        if pm:
            rest = (pm.group("rest") or "").strip()
            out_lines.append("Proof.")
            if rest:
                out_lines.append(rest)
            continue

        tm = THEOREM_MARK_RE.match(line)
        if tm:
            kind = (tm.group("kind") or "").title()
            num = (tm.group("num") or "").strip()
            rest = (tm.group("rest") or "").strip()
            out_lines.append(f"{kind} {num}.")
            if rest:
                out_lines.append(rest)
            continue

        out_lines.append(line)

    # avoid accidental triple blank lines
    txt = "\n".join(out_lines)
    txt = re.sub(r"\n{4,}", "\n\n\n", txt)
    return txt.strip()


# ---------------- model call & cleanup ----------------

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
    def repl(m: re.Match) -> str:
        inner = m.group(1).strip()
        return "\\[\n" + inner + "\n\\]"
    latex = re.sub(r"(?s)\$\$(.*?)\$\$", repl, latex)
    return latex


def normalize_unicode_symbols(latex: str) -> str:
    latex = latex.replace("§", r"\S ")
    latex = latex.replace("\u00A0", " ")
    return latex


@retry(
    reraise=True,
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    retry=retry_if_exception_type(Exception),
)
def markdown_to_latex(client: OpenAI, model: str, markdown: str, max_tokens: int) -> str:
    prompt = LATEX_CONVERT_PROMPT + markdown.strip()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    text = resp.choices[0].message.content or ""
    text = strip_code_fences(text)
    text = strip_outer_document(text)
    text = normalize_display_math(text)
    text = normalize_unicode_symbols(text)
    return text.strip()


# ---------------- sentinels ----------------

ENV_BLOCK_RE = re.compile(
    r"\\begin\{(?P<env>defn|thm|lem|prop|cor|proof)\}\s*(?P<body>.*?)\\end\{\1\}",
    re.DOTALL,
)

NUMBERED_TITLE_RE = re.compile(
    r"\b(Theorem|Lemma|Proposition|Corollary|Definition)\s+([0-9]+(?:\.[0-9]+)*)\s*\.?",
    re.IGNORECASE,
)

def _first_nonempty_line(s: str) -> str:
    for ln in s.splitlines():
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
    head = "\n".join(body.splitlines()[:8])
    head_plain = _strip_simple_latex_cmds(head)
    m = NUMBERED_TITLE_RE.search(head_plain)
    if m:
        kind = m.group(1).title()
        num = m.group(2)
        return f"{kind} {num}"
    line = _strip_simple_latex_cmds(_first_nonempty_line(body))
    if not line:
        return "UNKNOWN"
    if len(line) > 120:
        return line[:120] + "..."
    return line

def escape_attr(s: str) -> str:
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    return s

def insert_block_sentinels(latex: str) -> str:
    if "%<BLOCK" in latex:
        return latex
    out = []
    pos = 0
    for m in ENV_BLOCK_RE.finditer(latex):
        out.append(latex[pos:m.start()])
        env = m.group("env")
        body = m.group("body")
        label = extract_short_label(env, body)
        out.append(f'%<BLOCK type={env} label="{escape_attr(label)}">\n')
        out.append(m.group(0))
        out.append("\n%</BLOCK>\n")
        pos = m.end()
    out.append(latex[pos:])
    return "".join(out).strip()


# ---------------- document template ----------------

def build_tex_document(body_chunks: List[str]) -> str:
    preamble = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{geometry}
\geometry{margin=1in}

% Fixed theorem-like env names for downstream parsing
\newtheorem{thm}{Theorem}
\newtheorem{lem}[thm]{Lemma}
\newtheorem{prop}[thm]{Proposition}
\newtheorem{cor}[thm]{Corollary}
\theoremstyle{definition}
\newtheorem{defn}[thm]{Definition}

\begin{document}
"""
    end = r"""
\end{document}
"""
    return preamble + "\n\n".join(body_chunks).rstrip() + "\n" + end


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("in_md", type=str)
    ap.add_argument("out_tex", type=str)
    ap.add_argument("--workers", type=int, default=None, help="Override LLM worker count")
    args = ap.parse_args()

    cfg = load_config()
    settings = load_settings()
    api_key = require_str(cfg, "api_key")
    base_url = get_cfg(cfg, "base_url", "https://aihubmix.com/v1")
    model = require_str(cfg, "model")
    max_tokens_think = int(get_setting(settings, "MDTOTEX_MAX_TOKENS", 2048))

    in_md = Path(args.in_md).expanduser().resolve()
    out_tex = Path(args.out_tex).expanduser().resolve()
    if not in_md.exists():
        print(f"ERROR: Markdown not found: {in_md}", file=sys.stderr)
        sys.exit(2)

    workers: Optional[int] = args.workers
    if workers is None or workers <= 0:
        workers = int(get_setting(settings, "MDTOTEX_WORKERS", 4))

    md_text = in_md.read_text(encoding="utf-8")
    pages = split_markdown_pages(md_text)

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=int(get_setting(settings, "MDTOTEX_TIMEOUT", 120)),
    )

    body_chunks: List[str] = []
    last_top_section: Optional[str] = None

    # ---- (1) headings sequential (stable de-dup) + marker normalization ----
    page_meta: Dict[int, Tuple[List[str], str]] = {}
    for page_num, md in pages:
        md_norm = normalize_markdown_block_markers(md)
        heading_lines, remaining_md = extract_and_convert_headings(md_norm)

        # De-duplicate repeated top \\section across pages (rare for papers, but keep)
        filtered_headings: List[str] = []
        for h in heading_lines:
            if h.startswith(r"\section{"):
                if h == last_top_section:
                    continue
                last_top_section = h
            filtered_headings.append(h)

        page_meta[page_num] = (filtered_headings, remaining_md)

    # ---- (2) convert pages concurrently ----
    latex_by_page: Dict[int, str] = {}

    def _convert_one(page_num: int, remaining_md: str) -> str:
        latex = ""
        if remaining_md.strip():
            latex = markdown_to_latex(client, model, remaining_md, max_tokens=max_tokens_think)
            latex = insert_block_sentinels(latex)
        return latex

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_convert_one, page_num, page_meta[page_num][1]): page_num
            for page_num, _md in pages
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Convert pages"):
            page_num = futs[fut]
            latex_by_page[page_num] = fut.result()

    # ---- (3) assemble in original page order ----
    for page_num, _md in pages:
        filtered_headings, _remaining_md = page_meta[page_num]
        latex = latex_by_page.get(page_num, "")

        parts: List[str] = []
        parts.append(f"% ===== Page {page_num} =====")
        if filtered_headings:
            parts.append("\n".join(filtered_headings))
        if latex:
            parts.append(latex)
        # NOTE: no \\clearpage here (avoid forced breaks)

        body_chunks.append("\n".join(parts).strip())

    tex = build_tex_document(body_chunks)
    out_tex.write_text(tex, encoding="utf-8")
    print(f"DONE: {out_tex}")


if __name__ == "__main__":
    main()
