#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split LaTeX into JSON items (theorem-like envs + interleaving text),
following the schema:
[
  {
    "index": 1,
    "label": "Theorem 1",
    "env": "thm",
    "number_components": [1],
    "context": {"chapter":"", "section":"...", "chapter_number":null, "section_number":"2"},
    "content": "...",
    "dependencies": [],
    "proof": "..."
  },
  ...
]

Rules:
- theorem-like envs: theorem, lemma, corollary, remark, example
- Any non-theorem text between TWO theorem-like blocks becomes ONE item with env="text"
- Also keeps leading text before first theorem-like and trailing text after last theorem-like as env="text"
- Proof immediately following a theorem-like env (only whitespace between) is extracted into "proof".
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


THEOREM_ENVS = ["theorem", "lemma", "corollary", "remark", "example"]
ENV_TAG = {
    "theorem": "thm",
    "lemma": "lem",
    "corollary": "cor",
    "remark": "rem",
    "example": "ex",
}
LABEL_PREFIX = {
    "thm": "Theorem",
    "lem": "Lemma",
    "cor": "Corollary",
    "rem": "Remark",
    "ex": "Example",
}


@dataclass
class ContextState:
    section_title: str = ""
    section_number: Optional[str] = None
    chapter: str = ""
    chapter_number: Optional[str] = None


SECTION_RE = re.compile(r"\\(section|subsection|subsubsection)(\*)?\{([^}]*)\}")
DOCUMENT_BODY_RE = re.compile(r"\\begin\{document\}(.*)\\end\{document\}", re.S)

# theorem-like env blocks
ENV_RE = re.compile(
    r"\\begin\{(" + "|".join(THEOREM_ENVS) + r")\}(.*?)\\end\{\1\}",
    re.S,
)

# proof immediately after
PROOF_PREFIX_RE = re.compile(r"\s*\\begin\{proof\}(.*?)\\end\{proof\}", re.S)

LABEL_RE = re.compile(r"\\label\{([^}]+)\}")


def extract_body(tex: str) -> str:
    m = DOCUMENT_BODY_RE.search(tex)
    return m.group(1) if m else tex


def update_context_from_text(ctx: ContextState, text: str, sec_counter: List[int]) -> None:
    """
    Update ctx according to the LAST \\section{...} seen in `text`.
    We ignore numbering for starred sections (\\section*{...}).
    We treat subsection/subsubsection as part of section_title (append with " / ").
    """
    last_section_title = None
    last_section_starred = None
    last_sub_titles: List[str] = []

    for m in SECTION_RE.finditer(text):
        cmd, star, title = m.group(1), m.group(2), m.group(3)
        is_starred = (star == "*")

        if cmd == "section":
            last_section_title = title
            last_section_starred = is_starred
            last_sub_titles = []
        else:
            # keep as part of section title path
            last_sub_titles.append(title)

    if last_section_title is None and not last_sub_titles:
        return

    # if a new section appeared, update numbering if not starred
    if last_section_title is not None:
        ctx.section_title = last_section_title
        if last_section_starred:
            ctx.section_number = None
        else:
            sec_counter[0] += 1
            ctx.section_number = str(sec_counter[0])

    # append subsections to section title (path-like)
    if last_sub_titles:
        base = ctx.section_title or ""
        suffix = " / ".join(last_sub_titles)
        ctx.section_title = f"{base} / {suffix}" if base else suffix


def current_context_obj(ctx: ContextState) -> Dict[str, Any]:
    return {
        "chapter": ctx.chapter,
        "section": ctx.section_title,
        "chapter_number": ctx.chapter_number,
        "section_number": ctx.section_number,
    }


def label_from_env_block(env_name: str, env_tag: str, env_block: str, counters: Dict[str, int]) -> Tuple[str, List[int]]:
    """
    Prefer number extracted from \\label{thm:1} / \\label{lem:3} style.
    Otherwise fall back to per-env-tag running counters.
    """
    num: Optional[int] = None
    m = LABEL_RE.search(env_block)
    if m:
        key = m.group(1)
        # try suffix like ":12"
        mm = re.search(r":(\d+)$", key)
        if mm:
            num = int(mm.group(1))

    if num is None:
        counters[env_tag] += 1
        num = counters[env_tag]

    label = f"{LABEL_PREFIX[env_tag]} {num}"
    return label, [num]


def make_text_item(index: int, ctx: ContextState, content: str) -> Dict[str, Any]:
    return {
        "index": index,
        "label": "",
        "env": "text",
        "number_components": [],
        "context": current_context_obj(ctx),
        "content": content,
        "dependencies": [],
        "proof": "",
    }


def make_env_item(
    index: int,
    ctx: ContextState,
    env_name: str,
    env_block: str,
    proof_block: str,
    counters: Dict[str, int],
) -> Dict[str, Any]:
    env_tag = ENV_TAG[env_name]
    label, number_components = label_from_env_block(env_name, env_tag, env_block, counters)
    return {
        "index": index,
        "label": label,
        "env": env_tag,
        "number_components": number_components,
        "context": current_context_obj(ctx),
        "content": env_block,
        "dependencies": [],
        "proof": proof_block,
    }


def split_tex(tex: str) -> List[Dict[str, Any]]:
    body = extract_body(tex)

    ctx = ContextState()
    sec_counter = [0]  # mutable int
    env_counters = {tag: 0 for tag in LABEL_PREFIX.keys()}  # thm/lem/cor/rem/ex

    items: List[Dict[str, Any]] = []
    idx = 1
    pos = 0

    for m in ENV_RE.finditer(body):
        start, end = m.span()
        env_name = m.group(1)
        env_block = m.group(0)

        # pre-text
        pre = body[pos:start]
        update_context_from_text(ctx, pre, sec_counter)

        if pre.strip():
            # IMPORTANT: keep original order and original LaTeX (trim only outer newlines)
            items.append(make_text_item(idx, ctx, pre.strip("\n")))
            idx += 1

        # proof right after
        after = body[end:]
        pm = PROOF_PREFIX_RE.match(after)
        proof_block = ""
        proof_len = 0
        if pm:
            proof_block = "\\begin{proof}" + pm.group(1) + "\\end{proof}"
            proof_len = pm.end()

        items.append(
            make_env_item(
                idx,
                ctx,
                env_name,
                env_block.strip("\n"),
                proof_block.strip("\n"),
                env_counters,
            )
        )
        idx += 1

        pos = end + proof_len

    # tail text
    tail = body[pos:]
    update_context_from_text(ctx, tail, sec_counter)
    if tail.strip():
        items.append(make_text_item(idx, ctx, tail.strip("\n")))

    # Re-index strictly 1..N (just in case)
    for i, it in enumerate(items, start=1):
        it["index"] = i
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_tex", help="input .tex file")
    ap.add_argument("output_json", help="output .json file")
    args = ap.parse_args()

    tex = open(args.input_tex, "r", encoding="utf-8").read()
    items = split_tex(tex)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(items)} items -> {args.output_json}")


if __name__ == "__main__":
    main()
