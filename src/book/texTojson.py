#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert current book TeX (thm/proof + optional subpart sentinels) to JSON records.

Pipeline overview:
1) Parse top-level thm/proof blocks (outer sentinel first, env fallback).
2) Parse nested subpart sentinels inside each environment body.
3) Build per-exercise/per-part JSON rows:
   - thm -> problem
   - proof -> proof
4) For subparts, prepend shared prelude to every part body.

Validation layer (deterministic warnings):
- missing_proof
- unknown_exercise
- subpart_mismatch
- duplicate_source_idx

Optional enhance hooks (still deterministic by default):
- direct_answer extraction for value-type problems
- type refine (rule-based secondary pass)
- include raw tex snippets
- optional LLM self-check repair pass (disabled by default)

Expected output JSON block (per item):
{
  "index": 1,                      # int, contiguous from 1
  "problem": "...",                # str
  "proof": "...",                  # str
  "direct_answer": "",             # str (reserved)
  "题目类型": ["证明题"],             # list[str], one of: 证明题/求值题/其他
  "预估难度": [],                    # list (reserved)
  "source": "",                    # str (manual interface)
  "source_idx": "Exercise 2.8-(a)"# str, Exercise x.y or Exercise x.y-(part)
}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# -----------------------------
# User Fill Entry (edit here)
# -----------------------------
# Keep as an interface for manual filling.
SOURCE_NAME_ENTRY = ""


BLOCK_START_RE = re.compile(r"^\s*%<BLOCK\s+([^>]+)>\s*$")
BLOCK_END_RE = re.compile(r"^\s*%</BLOCK>\s*$")
KV_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=("(?:\\.|[^"])*"|\S+)')
EXERCISE_LABEL_RE = re.compile(r"\bExercise\s+(\d+(?:\.\d+)*)\b", re.IGNORECASE)
BEGIN_ENV_RE = re.compile(r"^\s*\\begin\{(thm|proof)\}\s*$")
END_ENV_RE = re.compile(r"^\s*\\end\{(thm|proof)\}\s*$")
SOURCE_IDX_RE = re.compile(
    r"^\s*Exercise\s+(?P<ex>\d+(?:\.\d+)*|UNKNOWN)(?:-\((?P<part>[^)]+)\))?\s*$",
    re.IGNORECASE,
)
TAG_RE = re.compile(r"\\tag\{([^}]+)\}")
TAG_TOKEN_RE = re.compile(r"\\tag\*?\{[^}]*\}")
MD_MANUAL_EQNUM_RE = re.compile(r"\(\s*(\d+\.\d+(?:\.\d+)*(?:[A-Za-z])?)\s*\)")
SAFE_EQREF_RE = re.compile(
    r"(?<!\\)\b(?:Eq\.|Eqs\.|Equation|Equations|problem|constraint|update|formula|model)\s*\(\s*(\d+(?:\.\d+)*(?:[A-Za-z])?)\s*\)",
    re.IGNORECASE,
)
BARE_TAG_RE = re.compile(r"(?<!\\)\((\d+(?:\.\d+)*(?:[A-Za-z])?)\)")
REFS_ITEM_RE = re.compile(r"(?i)^(theorem|thm|lemma|lem|algorithm|alg|proposition|prop|corollary|cor|definition|defn)\s*:\s*(\d+(?:\.\d+)*)$")
ORIGIN_BLOCK_RE = re.compile(r"(?ms)^%<BLOCK\s+([^>]*\btype=origintext\b[^>]*)>\s*\n(.*?)\n%</BLOCK>\s*")
THEOREM_LABEL_RE = re.compile(r"(?i)\b(?P<kind>Theorem|Thm\.?|Lemma|Lem\.?|Algorithm|Alg\.?)\s+(?P<num>\d+(?:\.\d+)*)\b")


_RETRYABLE_OPENAI_ERRORS = {"RateLimitError", "APITimeoutError", "APIError"}

# Type signal patterns used by rule classifier and conflict routing.
VALUE_TYPE_PATTERNS = [
    r"\bwhat is\b",
    r"\bcompute\b",
    r"\bfind\b",
    r"\bevaluate\b",
    r"\bcalculate\b",
    r"\bdistance\b",
    r"\bmaximum\b",
    r"\bminimum\b",
    r"\bmax\b",
    r"\bmin\b",
]

PROOF_TYPE_PATTERNS = [
    r"\bshow that\b",
    r"\bprove that\b",
    r"\bprove\b",
    r"\bshow\b",
    r"\bderive\b",
    r"\bestablish\b",
    r"\bif and only if\b",
    r"\biff\b",
    r"\bwhen does\b",  # often asks for conditions + justification
]


@dataclass
class Block:
    type: str
    label: str
    attrs: Dict[str, str] = field(default_factory=dict)
    items: List[Union[str, "Block"]] = field(default_factory=list)


def _unescape_attr(s: str) -> str:
    t = s
    if t.startswith('"') and t.endswith('"'):
        t = t[1:-1]
    t = t.replace("\\n", "\n")
    t = t.replace('\\"', '"')
    t = t.replace("\\\\", "\\")
    return t


def _parse_block_header(line: str) -> Optional[Tuple[str, str, Dict[str, str]]]:
    m = BLOCK_START_RE.match(line)
    if not m:
        return None
    body = m.group(1)

    attrs: Dict[str, str] = {}
    for km in KV_RE.finditer(body):
        k = km.group(1)
        v = _unescape_attr(km.group(2))
        attrs[k] = v

    btype = attrs.get("type", "").strip()
    label = attrs.get("label", "").strip()
    if not btype:
        return None
    return btype, label, attrs


def parse_subpart_items(lines: List[str]) -> List[Union[str, Block]]:
    """
    Parse only nested subpart sentinel blocks from an env body.
    Keeps surrounding lines as plain strings.
    """
    items: List[Union[str, Block]] = []
    stack: List[Block] = []

    for raw in lines:
        start = _parse_block_header(raw)
        if start is not None:
            btype, label, attrs = start
            # Only keep subpart_* blocks as structured children.
            # All other sentinels are treated as plain text.
            if btype.startswith("subpart_"):
                stack.append(Block(type=btype, label=label, attrs=attrs, items=[]))
                continue

        if BLOCK_END_RE.match(raw) and stack:
            node = stack.pop()
            if stack:
                stack[-1].items.append(node)
            else:
                items.append(node)
            continue

        if stack:
            stack[-1].items.append(raw)
        else:
            items.append(raw)

    return items


def parse_outer_blocks_from_sentinels(tex: str) -> List[Block]:
    """
    Parse all sentinel blocks and return root thm/proof blocks.
    Sentinel-first mode avoids re-inferring metadata that upstream already provided.
    """
    roots: List[Block] = []
    stack: List[Block] = []

    for raw in (tex or "").splitlines():
        start = _parse_block_header(raw)
        if start is not None:
            btype, label, attrs = start
            stack.append(Block(type=btype, label=label, attrs=attrs, items=[]))
            continue

        if BLOCK_END_RE.match(raw):
            if not stack:
                continue
            node = stack.pop()
            if stack:
                stack[-1].items.append(node)
            else:
                roots.append(node)
            continue

        if stack:
            stack[-1].items.append(raw)

    return [b for b in roots if b.type in {"thm", "proof"}]


def parse_outer_env_blocks(tex: str) -> List[Block]:
    """
    Parse top-level thm/proof environments directly from TeX.
    This is fallback when outer sentinel blocks are unavailable.
    """
    lines = (tex or "").splitlines()
    out: List[Block] = []
    i = 0
    while i < len(lines):
        m = BEGIN_ENV_RE.match(lines[i] or "")
        if not m:
            i += 1
            continue

        env = m.group(1)
        depth = 1
        j = i + 1
        body_lines: List[str] = []
        while j < len(lines):
            ln = lines[j]
            mb = BEGIN_ENV_RE.match(ln or "")
            me = END_ENV_RE.match(ln or "")
            if mb and mb.group(1) == env:
                depth += 1
            if me and me.group(1) == env:
                depth -= 1
                if depth == 0:
                    break
            body_lines.append(ln)
            j += 1

        # Derive a lightweight label from visible content.
        # For thm this is typically "Exercise x.y"; proof uses fixed label.
        label = ""
        if env == "thm":
            for ln in body_lines:
                t = (ln or "").strip()
                if not t:
                    continue
                m_ex = EXERCISE_LABEL_RE.search(t)
                if m_ex:
                    label = f"Exercise {m_ex.group(1)}"
                    break
        else:
            label = "Proof"

        out.append(Block(type=env, label=label, attrs={}, items=parse_subpart_items(body_lines)))
        i = j + 1 if j < len(lines) else j

    return out


def _items_to_text(items: List[Union[str, Block]], *, keep_children: bool = False) -> str:
    out: List[str] = []
    for it in items:
        if isinstance(it, str):
            out.append(it)
        elif keep_children:
            out.append(_block_to_text(it))
    return "\n".join(out).strip()


def _strip_outer_env_lines(text: str) -> str:
    lines = (text or "").splitlines()
    kept: List[str] = []
    for ln in lines:
        if BEGIN_ENV_RE.match(ln) or END_ENV_RE.match(ln):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def _block_to_text(block: Block) -> str:
    txt = _items_to_text(block.items, keep_children=False)
    return _strip_outer_env_lines(txt).strip()


def _extract_exercise_num(label_or_text: str) -> str:
    m = EXERCISE_LABEL_RE.search(label_or_text or "")
    return (m.group(1) if m else "").strip()


def _compose_with_common_prefix(common_prefix: str, subpart_text: str) -> str:
    a = (common_prefix or "").strip()
    b = (subpart_text or "").strip()
    # Avoid duplicating the same stem when an upstream stage already injected it.
    if a and b and b.startswith(a):
        return b
    if a and b:
        return a + "\n\n" + b
    return a or b


def _split_prelude_and_subparts(block: Block, subpart_type: str) -> Tuple[str, List[Block]]:
    prelude_lines: List[str] = []
    subparts: List[Block] = []

    for it in block.items:
        if isinstance(it, str):
            prelude_lines.append(it)
        elif isinstance(it, Block):
            if it.type == subpart_type:
                subparts.append(it)
            else:
                # Defensive fallback: unknown nested block is flattened into text,
                # so no source content is silently dropped.
                prelude_lines.append(_block_to_text(it))

    prelude = _strip_outer_env_lines("\n".join(prelude_lines)).strip()
    return prelude, subparts


def _exercise_from_subparts(subparts: List[Block]) -> str:
    """
    Infer exercise number from nested subpart sentinel attrs.
    Used for multi-solution layout where per-part thm/proof may not include
    an explicit 'Exercise x.y' line in visible text.
    """
    vals: List[str] = []
    for sp in subparts:
        ex = (sp.attrs.get("exercise") or "").strip()
        if ex:
            vals.append(ex)
    if not vals:
        return ""
    # Keep first; upstream emitter should keep exercise attr consistent.
    return vals[0]


def _source_idx(exercise_num: str, part: str) -> str:
    if exercise_num and part:
        return f"Exercise {exercise_num}-({part})"
    if exercise_num:
        return f"Exercise {exercise_num}"
    if part:
        return f"Exercise UNKNOWN-({part})"
    return "Exercise UNKNOWN"



def _merge_ordered_unique(primary: List[str], secondary: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for x in list(primary or []) + list(secondary or []):
        t = (x or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _normalize_statement_kind(kind_raw: str) -> str:
    t = (kind_raw or "").strip().lower().rstrip(".")
    if t in {"lemma", "lem"}:
        return "lemma"
    if t in {"algorithm", "alg"}:
        return "algorithm"
    return "theorem"


def _theorem_label_key_from_ref(ref_key: str) -> str:
    t = (ref_key or "").strip()
    m = re.match(r"^(theorem|lemma|algorithm):(\d+(?:\.\d+)*)$", t, re.IGNORECASE)
    if not m:
        return f"thm:{t}" if t else ""
    kind = _normalize_statement_kind(m.group(1))
    num = (m.group(2) or "").strip()
    if kind == "lemma":
        return f"lem:{num}"
    if kind == "algorithm":
        return f"alg:{num}"
    return f"thm:{num}"


def _parse_theorem_refs_attr(refs_attr: str) -> List[str]:
    vals: List[str] = []
    if not refs_attr:
        return vals
    parts = re.split(r"[;,]", refs_attr)
    for raw in parts:
        t = (raw or "").strip()
        if not t:
            continue
        m = REFS_ITEM_RE.match(t)
        if not m:
            continue
        kind = _normalize_statement_kind(m.group(1) or "")
        num = (m.group(2) or "").strip()
        if num:
            vals.append(f"{kind}:{num}")
    return _merge_ordered_unique([], vals)


def _extract_theorem_ref_from_label(label: str) -> str:
    m = THEOREM_LABEL_RE.search(label or "")
    if not m:
        return ""
    kind = _normalize_statement_kind(m.group("kind") or "")
    num = (m.group("num") or "").strip()
    return f"{kind}:{num}" if num else ""


def build_doc_theorem_index(tex: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    txt = tex or ""
    for m in ORIGIN_BLOCK_RE.finditer(txt):
        hdr = m.group(1) or ""
        body = (m.group(2) or "").strip()
        attrs: Dict[str, str] = {}
        for km in KV_RE.finditer(hdr):
            k = km.group(1)
            v = _unescape_attr(km.group(2))
            attrs[k] = v
        label = (attrs.get("label") or "").strip()
        ref_key = _extract_theorem_ref_from_label(label)
        if not ref_key:
            ref_key = _extract_theorem_ref_from_label("\n".join(body.splitlines()[:6]))
        if not ref_key or ref_key in out:
            continue
        kind, num = ref_key.split(":", 1)
        if kind == "lemma":
            display_kind = "Lemma"
        elif kind == "algorithm":
            display_kind = "Algorithm"
        else:
            display_kind = "Theorem"
        label_key = _theorem_label_key_from_ref(ref_key)
        snippet = re.sub(r"\s+", " ", body).strip()
        out[ref_key] = {
            "theorem": num,
            "label_key": label_key,
            "display_ref": f"{display_kind} {num}",
            "kind": kind,
            "theorem_content": snippet,
            "content": snippet,
        }
    return out

def _normalize_tag_token(s: str) -> str:
    t = (s or "").strip()
    t = t.strip("()[]")
    t = re.sub(r"\s+", "", t)
    return t


def extract_equation_context(tex: str, tag_match_start: int, window: int = 240) -> str:
    left = max(0, int(tag_match_start) - int(window))
    right = min(len(tex or ""), int(tag_match_start) + int(window))
    snippet = (tex or "")[left:right]
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return snippet


def _extract_tagged_equation_block(tex: str, tag: str) -> str:
    """
    Try to extract only the display-math block that contains \\tag{tag}.
    Falls back to empty string when no clear block is found.
    """
    if not tex or not tag:
        return ""
    t = _normalize_tag_token(tag)
    if not t:
        return ""
    tag_pat = re.escape(t)

    # 1) \[ ... \tag{t} ... \]
    p1 = re.compile(
        rf"(?s)\\\[(?:(?!\\\]).)*?\\tag\{{\s*{tag_pat}\s*\}}(?:(?!\\\]).)*?\\\]"
    )
    m1 = p1.search(tex)
    if m1:
        return (m1.group(0) or "").strip()

    # 2) equation/align-like envs that contain \tag{t}
    p2 = re.compile(
        rf"(?s)\\begin\{{(?P<env>equation\*?|align\*?|gather\*?|multline\*?)\}}"
        rf"(?:(?!\\end\{{(?P=env)\}}).)*?\\tag\{{\s*{tag_pat}\s*\}}"
        rf"(?:(?!\\end\{{(?P=env)\}}).)*?\\end\{{(?P=env)\}}"
    )
    m2 = p2.search(tex)
    if m2:
        return (m2.group(0) or "").strip()

    return ""




def _standardize_equation_preview_tex(block: str) -> str:
    """
    Normalize extracted equation snippets for preview.
    Prefer align* when a bracket display contains aligned + multiple tags.
    """
    b = (block or "").strip()
    if not b:
        return ""

    if re.match(r"(?s)^\\begin\{align\*?\}.*\\end\{align\*?\}$", b):
        return b

    m_br = re.match(r"(?s)^\\\[(?P<inner>.*)\\\]$", b)
    if not m_br:
        return b

    inner = (m_br.group("inner") or "").strip()
    tags = [t.strip() for t in TAG_TOKEN_RE.findall(inner) if t.strip()]

    trailing_num_re = re.compile(
        r"(?s)^(?P<body>.*?)(?:(?:\s*(?:\\qquad|\\quad|\\hfill)\s*)+|\s+)\((?P<num>\d+(?:\.\d+)*(?:[A-Za-z])?)\)\s*$"
    )

    def _split_row_manual_num(row: str) -> Tuple[str, Optional[str]]:
        rr = (row or "").strip()
        if not rr:
            return "", None
        m = trailing_num_re.match(rr)
        if not m:
            return rr, None
        return (m.group("body") or "").rstrip(), (m.group("num") or "").strip()

    m_al = re.search(r"(?s)\\begin\{aligned\}(?P<rows>.*?)\\end\{aligned\}", inner)
    if m_al:
        aligned_rows = (m_al.group("rows") or "").strip()
        rows = [r.strip() for r in re.split(r"\\\\\s*", aligned_rows) if r.strip()]
        if rows:
            out_rows: List[str] = []
            has_any_tag = False
            for i, r in enumerate(rows):
                r0 = TAG_TOKEN_RE.sub("", r).strip()
                core, manual_num = _split_row_manual_num(r0)
                row_tag = None
                if i < len(tags):
                    row_tag = tags[i]
                elif manual_num:
                    row_tag = rf"\\tag{{{manual_num}}}"
                if row_tag:
                    has_any_tag = True
                    out_rows.append((core + " " + row_tag).strip())
                else:
                    out_rows.append(core)
            if has_any_tag:
                return "\\begin{align*}\n" + " \\\\\n".join(out_rows) + "\n\\end{align*}"

    if len(tags) > 1:
        core = TAG_TOKEN_RE.sub("", inner).strip()
        rows = [r.strip() for r in re.split(r"\\\\\s*", core) if r.strip()]
        if rows:
            out_rows: List[str] = []
            for i, r in enumerate(rows):
                rc, manual_num = _split_row_manual_num(r)
                if i < len(tags):
                    out_rows.append((rc + " " + tags[i]).strip())
                elif manual_num:
                    out_rows.append((rc + " " + rf"\\tag{{{manual_num}}}").strip())
                else:
                    out_rows.append(rc)
            return "\\begin{align*}\n" + " \\\\\n".join(out_rows) + "\n\\end{align*}"

    return b




def _extract_single_tag_preview_tex(block: str, tag: str) -> str:
    """
    Extract a minimal preview snippet for one target tag from a standardized equation block.
    Returns a tiny align* block when possible; otherwise returns the original block.
    """
    b = (block or "").strip()
    t = _normalize_tag_token(tag)
    if not b or not t:
        return b

    tag_pat = re.compile(rf"\\tag\*?\{{\s*{re.escape(t)}\s*\}}")

    m_al = re.match(r"(?s)^\\begin\{align\*?\}(?P<body>.*)\\end\{align\*?\}$", b)
    if m_al:
        body = (m_al.group("body") or "").strip()
        rows = [r.strip() for r in re.split(r"\\\\\s*", body) if r.strip()]
        for r in rows:
            if tag_pat.search(r):
                return "\\begin{align*}\n" + r + "\n\\end{align*}"
        return b

    # For non-align blocks (single-equation), keep as-is.
    return b

def build_doc_tag_index(tex: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for m in TAG_RE.finditer(tex or ""):
        tag = _normalize_tag_token(m.group(1) or "")
        if not tag or tag in out:
            continue
        eq_full = _standardize_equation_preview_tex(_extract_tagged_equation_block(tex, tag))
        eq_single = _extract_single_tag_preview_tex(eq_full, tag)
        out[tag] = {
            "tag": tag,
            "label_key": f"eq:{tag}",
            "display_tag": f"({tag})",
            "kind": "equation",
            "content": extract_equation_context(tex, m.start()),
            "equation_content": eq_single,
            "equation_content_full": eq_full,
            "position": m.start(),
        }
    return out


def _iter_display_math_blocks(text: str) -> List[Tuple[int, str]]:
    """
    Extract display-math blocks with position.
    Supports bracket displays and common amsmath environments.
    """
    txt = text or ""
    spans: List[Tuple[int, int, str]] = []

    p_br = re.compile(r"(?s)\\\[(?P<body>.*?)\\\]")
    p_dollar = re.compile(r"(?s)\$\$(?P<body>.*?)\$\$")
    p_env = re.compile(
        r"(?s)\\begin\{(?P<env>equation\*?|align\*?|gather\*?|multline\*?)\}"
        r"(?P<body>.*?)\\end\{(?P=env)\}"
    )

    for m in p_br.finditer(txt):
        spans.append((m.start(), m.end(), (m.group(0) or "").strip()))
    for m in p_dollar.finditer(txt):
        spans.append((m.start(), m.end(), (m.group(0) or "").strip()))
    for m in p_env.finditer(txt):
        spans.append((m.start(), m.end(), (m.group(0) or "").strip()))

    spans.sort(key=lambda x: (x[0], x[1]))
    out: List[Tuple[int, str]] = []
    seen: Set[Tuple[int, int]] = set()
    for st, ed, blk in spans:
        key = (st, ed)
        if key in seen:
            continue
        seen.add(key)
        if blk:
            out.append((st, blk))
    return out


def build_md_tag_fallback_index(md_text: str) -> Dict[str, Dict[str, Any]]:
    """
    Build an equation-reference index from Markdown display blocks.
    Used as fallback when TeX has no corresponding \tag.

    Safety policy:
    - If a block contains explicit \tag, trust only those tags.
    - If the same tag maps to multiple different blocks, drop this tag as ambiguous.
      (Prefer missing mapping over wrong mapping.)
    """
    out: Dict[str, Dict[str, Any]] = {}
    txt = (md_text or "").replace("（", "(").replace("）", ")")
    if not txt.strip():
        return out

    candidates: Dict[str, List[Dict[str, Any]]] = {}

    for pos, blk in _iter_display_math_blocks(txt):
        blk0 = (blk or "").strip()
        if not blk0:
            continue

        tagged = [_normalize_tag_token(x) for x in TAG_RE.findall(blk0)]
        manual = [_normalize_tag_token(x) for x in MD_MANUAL_EQNUM_RE.findall(blk0)]

        # If explicit \tag exists in this block, trust only those tags.
        tags = _merge_ordered_unique(tagged, []) if tagged else _merge_ordered_unique([], manual)

        # Base aliases: if block has 18.50a/18.50b, also map 18.50 to same block.
        bases: List[str] = []
        for tg in tags:
            m_base = re.match(r"^(\d+\.\d+(?:\.\d+)*)([A-Za-z])$", tg)
            if m_base:
                bases.append(m_base.group(1))
        tags = _merge_ordered_unique(tags, bases)

        if not tags:
            continue

        eq_full = _standardize_equation_preview_tex(blk0)
        for tag in tags:
            if not tag:
                continue
            eq_single = eq_full
            if "\tag" in eq_full:
                eq_single = _extract_single_tag_preview_tex(eq_full, tag)

            rec = {
                "tag": tag,
                "label_key": f"eq:{tag}",
                "display_tag": f"({tag})",
                "kind": "equation",
                "content": re.sub(r"\s+", " ", blk0).strip(),
                "equation_content": eq_single,
                "equation_content_full": eq_full,
                "position": pos,
                "ref_source": "md_fallback",
            }
            candidates.setdefault(tag, []).append(rec)

    for tag, arr in candidates.items():
        if not arr:
            continue

        # de-dup by normalized equation full text
        by_eq: Dict[str, Dict[str, Any]] = {}
        for rec in arr:
            key = "\n".join([ln.rstrip() for ln in str(rec.get("equation_content_full") or rec.get("equation_content") or "").splitlines()]).strip()
            if key not in by_eq:
                by_eq[key] = rec

        # Ambiguous tag -> skip to avoid wrong mapping.
        if len(by_eq) > 1:
            continue

        out[tag] = next(iter(by_eq.values()))

    return out


def extract_defined_tags(raw_tex: str) -> List[str]:
    seen: Set[str] = set()
    vals: List[str] = []
    for m in TAG_RE.finditer(raw_tex or ""):
        tag = _normalize_tag_token(m.group(1) or "")
        if tag and tag not in seen:
            seen.add(tag)
            vals.append(tag)
    return vals


def extract_body_refs(
    raw_tex: str,
    known_tags: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], Dict[str, str], List[str]]:
    txt = raw_tex or ""

    # Normalize common TeX variants in references, e.g. (18.50\mathrm{b}) -> (18.50b)
    tnorm = txt
    tnorm = re.sub(r"\\(?:mathrm|text|textrm)\{\s*([A-Za-z])\s*\}", r"\1", tnorm)
    tnorm = tnorm.replace("（", "(").replace("）", ")")
    tnorm = tnorm.replace("–", "-").replace("—", "-")

    refs: List[str] = []
    evidence: Dict[str, str] = {}
    unresolved: List[str] = []
    seen: Set[str] = set()
    md_suffix_by_base: Dict[str, List[str]] = {}

    # Build md-fallback base aliases, e.g. 14.44 -> [14.44a, 14.44b, ...].
    for kt, rec in (known_tags or {}).items():
        m_suffix = re.match(r"^(\d+(?:\.\d+)*)([A-Za-z])$", (kt or ""))
        if not m_suffix:
            continue
        if str((rec or {}).get("ref_source") or "") != "md_fallback":
            continue
        base = m_suffix.group(1)
        md_suffix_by_base.setdefault(base, []).append(kt)
    for base in list(md_suffix_by_base.keys()):
        md_suffix_by_base[base] = sorted(
            _merge_ordered_unique([], md_suffix_by_base[base]),
            key=lambda x: x.lower(),
        )

    num_tok = r"\d+(?:\.\d+)*(?:[A-Za-z])?"
    punct_re = re.compile(rf"\((?P<tag>{num_tok})\)")
    range_re = re.compile(
        rf"\((?P<lbase>\d+(?:\.\d+)*)(?P<lch>[A-Za-z])\)\s*(?:--|-|to|~)\s*\((?P<rbase>\d+(?:\.\d+)*)(?P<rch>[A-Za-z])\)",
        re.IGNORECASE,
    )

    def _emit(tag_raw: str, ev: str) -> None:
        tag = _normalize_tag_token(tag_raw)
        if not tag or tag in seen:
            return
        if tag in known_tags:
            seen.add(tag)
            refs.append(tag)
            evidence[tag] = ev.strip()
            return

        # If base tag is missing, expand to md fallback suffixed tags:
        # (14.44) -> 14.44a/14.44b/... when available.
        m_base = re.match(r"^(\d+(?:\.\d+)*)$", tag)
        if m_base:
            base = m_base.group(1)
            expanded = md_suffix_by_base.get(base, [])
            for tg in expanded:
                if tg in seen:
                    continue
                seen.add(tg)
                refs.append(tg)
                evidence[tg] = ev.strip()
            if expanded:
                return

        unresolved.append(tag)

    # 1) phrase-scoped refs (Eq./Equation/problem/constraint ...)
    for m in SAFE_EQREF_RE.finditer(tnorm):
        _emit((m.group(1) or ""), (m.group(0) or ""))

    # 2) ranged refs, e.g. (18.50b)--(18.50e)
    for m in range_re.finditer(tnorm):
        lbase = (m.group("lbase") or "").strip()
        rbase = (m.group("rbase") or "").strip()
        lch = (m.group("lch") or "").strip().lower()
        rch = (m.group("rch") or "").strip().lower()
        ev = (m.group(0) or "")
        if lbase and rbase and lbase == rbase and lch.isalpha() and rch.isalpha():
            a, b = ord(lch), ord(rch)
            if a <= b:
                for c in range(a, b + 1):
                    _emit(f"{lbase}{chr(c)}", ev)
            else:
                for c in range(a, b - 1, -1):
                    _emit(f"{lbase}{chr(c)}", ev)

    # 3) generic bracket refs (18.17), (18.55b)
    for m in punct_re.finditer(tnorm):
        _emit((m.group("tag") or ""), (m.group(0) or ""))

    # 4) keep legacy bare matcher for compatibility
    for m in BARE_TAG_RE.finditer(tnorm):
        _emit((m.group(1) or ""), (m.group(0) or ""))

    # Stable unresolved ordering.
    unresolved = _merge_ordered_unique([], unresolved)
    return refs, evidence, unresolved


def compose_problem_with_context(
    problem: str,
    eq_targets: List[Dict[str, Any]],
    theorem_targets: Optional[List[Dict[str, Any]]] = None,
) -> str:
    p = (problem or "").strip()
    theorem_targets = theorem_targets or []

    chunks: List[str] = []

    if eq_targets:
        if len(eq_targets) == 1:
            t = eq_targets[0]
            eq_text = (t.get("equation_content") or t.get("content") or "").strip()
            chunks.append(f"Reference formula {t.get('display_tag', '')}: {eq_text}.")
        else:
            refs: List[str] = []
            for t in eq_targets:
                eq_text = (t.get("equation_content") or t.get("content") or "").strip()
                refs.append(f"{t.get('display_tag', '')} {eq_text}".strip())
            chunks.append("Reference formulas: " + "; ".join(refs) + ".")

    if theorem_targets:
        if len(theorem_targets) == 1:
            t = theorem_targets[0]
            ttxt = (t.get("theorem_content") or t.get("content") or "").strip()
            chunks.append(f"Reference result {t.get('display_ref', '')}: {ttxt}.")
        else:
            refs: List[str] = []
            for t in theorem_targets:
                ttxt = (t.get("theorem_content") or t.get("content") or "").strip()
                refs.append(f"{t.get('display_ref', '')} {ttxt}".strip())
            chunks.append("Reference results: " + "; ".join(refs) + ".")

    if not chunks:
        return p
    return (" ".join(chunks) + " Problem: " + p).strip()


def _normalize_problem_text(problem_text: str) -> str:
    s = (problem_text or "").lower()
    s = re.sub(r"\\[a-zA-Z]+", " ", s)  # drop TeX command noise for keyword matching
    s = re.sub(r"\s+", " ", s)
    return s


def _problem_type_signal_flags(problem_text: str) -> Tuple[bool, bool]:
    """
    Returns (value_hit, proof_hit) from deterministic keyword rules.
    """
    s = _normalize_problem_text(problem_text)
    value_hit = any(re.search(p, s) for p in VALUE_TYPE_PATTERNS)
    proof_hit = any(re.search(p, s) for p in PROOF_TYPE_PATTERNS)
    return value_hit, proof_hit


def infer_problem_type(problem_text: str) -> str:
    """
    Classify into one of:
    - 证明题
    - 求值题
    - 其他

    Rule priority:
    1) value/computation intent -> 求值题
    2) proof/show intent -> 证明题
    3) fallback -> 其他
    """
    value_hit, proof_hit = _problem_type_signal_flags(problem_text)

    # 求值题: direct numeric/closed-form computation style prompts.
    if value_hit:
        return "求值题"

    # 证明题: show/prove/derive/establish style prompts.
    if proof_hit:
        return "证明题"

    return "其他"


def refine_problem_type(problem_text: str, proof_text: str, initial_type: str, *, enable: bool) -> str:
    """
    Optional second-stage refinement (rule-based).
    Keeps default deterministic behavior when disabled.
    """
    if not enable:
        return initial_type
    s = (problem_text or "").lower()
    if initial_type == "其他":
        if re.search(r"\b(show|prove|derive|if and only if|when does)\b", s):
            return "证明题"
        if re.search(r"\b(what is|compute|find|evaluate|calculate|max|min|distance)\b", s):
            return "求值题"
    return initial_type


def extract_direct_answer(problem_type: str, proof_text: str) -> str:
    """
    Conservative direct-answer extractor for value problems.
    Returns empty string when no stable signal is found.
    """
    if problem_type != "求值题":
        return ""
    t = (proof_text or "").strip()
    if not t:
        return ""

    # Prefer last display-math block as a compact closed-form answer candidate.
    disp = re.findall(r"(?s)\\\[(.*?)\\\]", t)
    if disp:
        return "\\[\n" + disp[-1].strip() + "\n\\]"

    # Fallback: last sentence with conclusion cue.
    concl = re.findall(r"(?im)(?:therefore|thus|hence|so)\b[^\n]*", t)
    if concl:
        return "\\[\n" + concl[-1].strip() + "\n\\]"

    # Last non-empty line fallback.
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return ""
    return "\\[\n" + lines[-1][:300] + "\n\\]"


def build_records_from_tex(
    tex: str,
    *,
    source_name: str = "",
    include_raw: bool = False,
    enable_type_refine: bool = False,
    enable_direct_answer: bool = True,
    md_text: str = "",
) -> Tuple[List[Dict], List[Dict]]:
    sent_roots = parse_outer_blocks_from_sentinels(tex)
    env_roots = parse_outer_env_blocks(tex)
    # Block-first with coverage guard:
    # if outer BLOCK roots look incomplete, fallback to env parse to avoid silent data loss.
    use_sent = bool(sent_roots)
    if use_sent and env_roots:
        # Require sentinel coverage to be reasonably close to env coverage.
        use_sent = len(sent_roots) >= max(2, int(len(env_roots) * 0.6))
    roots = sent_roots if use_sent else env_roots
    if use_sent and env_roots:
        # Sentinel-first can miss valid thm/proof env blocks when outer BLOCK wrappers
        # are absent for a subset of exercises. Merge missing exercise segments from env parse.
        sent_ex: Set[str] = set()
        for b in sent_roots:
            if b.type != "thm":
                continue
            ex = _extract_exercise_num(b.label) or _extract_exercise_num(_block_to_text(b))
            if ex:
                sent_ex.add(ex)

        extra_env: List[Block] = []
        i = 0
        while i < len(env_roots):
            b = env_roots[i]
            if b.type != "thm":
                i += 1
                continue
            ex = _extract_exercise_num(b.label) or _extract_exercise_num(_block_to_text(b))
            if ex and ex not in sent_ex:
                extra_env.append(b)
                j = i + 1
                while j < len(env_roots) and env_roots[j].type != "thm":
                    extra_env.append(env_roots[j])
                    j += 1
                i = j
                continue
            i += 1
        if extra_env:
            roots = roots + extra_env

    # Key: (exercise_num, part_id) where part_id is "" for non-subpart entries.
    records_map: Dict[Tuple[str, str], Dict[str, str]] = {}
    exercise_common_problem_prefix: Dict[str, str] = {}
    last_exercise_num = ""
    # For fallback proof pairing in multi-solution streams without subpart_proof tags.
    pending_problem_parts_by_ex: Dict[str, List[str]] = {}
    warnings: List[Dict] = []

    def ensure_slot(ex_num: str, part_id: str) -> Dict[str, str]:
        # Central slot allocator to merge statement/proof from different passes.
        key = (ex_num, part_id)
        if key not in records_map:
            records_map[key] = {
                "exercise": ex_num,
                "part": part_id,
                "problem": "",
                "proof": "",
                "raw_problem_tex": "",
                "raw_proof_tex": "",
                "theorem_refs": [],
                "theorem_ref_evidence": {},
            }
        return records_map[key]

    def _statement_parts_for_ex(ex_num: str) -> List[str]:
        parts: List[str] = []
        for (ex, part), row in records_map.items():
            if ex != ex_num or not part:
                continue
            if (row.get("problem") or "").strip():
                parts.append(part)
        parts = sorted(set(parts), key=lambda p: (len(p) != 1 or not p.isalpha(), p))
        return parts

    def _proof_parts_for_ex(ex_num: str) -> List[str]:
        parts: List[str] = []
        for (ex, part), row in records_map.items():
            if ex != ex_num or not part:
                continue
            if (row.get("proof") or "").strip():
                parts.append(part)
        parts = sorted(set(parts), key=lambda p: (len(p) != 1 or not p.isalpha(), p))
        return parts

    for blk in roots:
        if blk.type == "thm":
            # Problem side: thm body (or per-subpart statement bodies).
            body_text = _block_to_text(blk)
            ex_num = _extract_exercise_num(blk.label) or _extract_exercise_num(body_text)
            prelude, subparts = _split_prelude_and_subparts(blk, "subpart_statement")
            # Multi-solution mode often emits per-part theorem blocks without explicit
            # "Exercise x.y" line; recover exercise from subpart attrs.
            if not ex_num and subparts:
                ex_num = _exercise_from_subparts(subparts)
            if not ex_num:
                ex_num = last_exercise_num
            if ex_num:
                last_exercise_num = ex_num

            if ex_num and prelude:
                old = exercise_common_problem_prefix.get(ex_num, "")
                if len(prelude) > len(old):
                    exercise_common_problem_prefix[ex_num] = prelude
            if subparts:
                for sp in subparts:
                    part_id = (sp.attrs.get("part") or "").strip().lower()
                    ex_sp = (sp.attrs.get("exercise") or "").strip() or ex_num
                    slot_ex = ex_sp or ex_num
                    slot = ensure_slot(slot_ex, part_id)
                    sp_refs = _parse_theorem_refs_attr((sp.attrs.get("refs") or "").strip())
                    if sp_refs:
                        slot["theorem_refs"] = _merge_ordered_unique(list(slot.get("theorem_refs") or []), sp_refs)
                        ev = dict(slot.get("theorem_ref_evidence") or {})
                        for t in sp_refs:
                            ev.setdefault(t, "refs_attr")
                        slot["theorem_ref_evidence"] = ev
                    sp_text = _block_to_text(sp)
                    # Keep per-subpart statement pure; shared prelude is migrated in final pass.
                    slot["problem"] = sp_text
                    slot["raw_problem_tex"] = sp_text
                    if slot_ex and part_id:
                        pend = pending_problem_parts_by_ex.setdefault(slot_ex, [])
                        if part_id not in pend:
                            pend.append(part_id)
            else:
                slot = ensure_slot(ex_num, "")
                blk_refs = _parse_theorem_refs_attr((blk.attrs.get("refs") or "").strip())
                if blk_refs:
                    slot["theorem_refs"] = _merge_ordered_unique(list(slot.get("theorem_refs") or []), blk_refs)
                    ev = dict(slot.get("theorem_ref_evidence") or {})
                    for t in blk_refs:
                        ev.setdefault(t, "refs_attr")
                    slot["theorem_ref_evidence"] = ev
                slot["problem"] = body_text
                slot["raw_problem_tex"] = body_text

        elif blk.type == "proof":
            # Proof side: proof body (or per-subpart proof bodies).
            prelude, subparts = _split_prelude_and_subparts(blk, "subpart_proof")
            if subparts:
                parts_seen_by_ex: Dict[str, List[str]] = {}
                for sp in subparts:
                    ex_num = (sp.attrs.get("exercise") or "").strip() or last_exercise_num
                    part_id = (sp.attrs.get("part") or "").strip().lower()
                    # Heuristic repair: if proof part id is duplicated but statement has
                    # missing parts, map this duplicate proof to the first missing statement part.
                    if ex_num and part_id:
                        stmt_parts = _statement_parts_for_ex(ex_num)
                        proof_parts = _proof_parts_for_ex(ex_num)
                        if part_id in proof_parts:
                            missing_stmt = [p for p in stmt_parts if p not in proof_parts]
                            if missing_stmt:
                                part_id = missing_stmt[0]
                    slot = ensure_slot(ex_num, part_id)
                    sp_text = _block_to_text(sp)
                    # Keep subpart proof pure; prelude is assigned separately.
                    slot["proof"] = sp_text
                    slot["raw_proof_tex"] = sp_text
                    if ex_num and part_id:
                        parts_seen_by_ex.setdefault(ex_num, [])
                        if part_id not in parts_seen_by_ex[ex_num]:
                            parts_seen_by_ex[ex_num].append(part_id)

                # If proof prelude exists, assign it to (a) or the first missing statement part.
                pre = (prelude or "").strip()
                if pre:
                    for ex_num, proof_parts in parts_seen_by_ex.items():
                        stmt_parts = _statement_parts_for_ex(ex_num)
                        if not stmt_parts:
                            continue
                        proof_set = set(proof_parts)
                        target = ""
                        if "a" in stmt_parts and "a" not in proof_set:
                            target = "a"
                        else:
                            for p in stmt_parts:
                                if p not in proof_set:
                                    target = p
                                    break
                        if not target:
                            continue
                        slot = ensure_slot(ex_num, target)
                        existing = (slot.get("proof") or "").strip()
                        merged = _compose_with_common_prefix(pre, existing) if existing else pre
                        slot["proof"] = merged
                        slot["raw_proof_tex"] = merged
            else:
                ex_num = last_exercise_num
                proof_text = _block_to_text(blk)
                # Multi-solution fallback: when proof block has no subpart tags,
                # map it to the earliest pending subpart for current exercise.
                pend = pending_problem_parts_by_ex.get(ex_num or "", [])
                if ex_num and pend:
                    part_id = pend.pop(0)
                    slot = ensure_slot(ex_num, part_id)
                else:
                    slot = ensure_slot(ex_num, "")
                slot["proof"] = proof_text
                slot["raw_proof_tex"] = proof_text

    def _sort_key(item: Tuple[Tuple[str, str], Dict[str, str]]):
        (ex_num, part_id), _v = item
        ex_parts: List[int] = []
        for p in (ex_num or "").split("."):
            ex_parts.append(int(p) if p.isdigit() else 10**9)
        part_rank = 10**6
        if part_id:
            if len(part_id) == 1 and part_id.isalpha():
                part_rank = ord(part_id.lower()) - ord("a")
            else:
                part_rank = 10**5
        return ex_parts, part_rank, ex_num, part_id

    ordered = [v for _k, v in sorted(records_map.items(), key=_sort_key)]
    doc_tag_index = build_doc_tag_index(tex)
    md_tag_index = build_md_tag_fallback_index(md_text)
    all_tag_index: Dict[str, Dict[str, Any]] = dict(doc_tag_index)
    for k, v in md_tag_index.items():
        all_tag_index.setdefault(k, v)
    doc_theorem_index = build_doc_theorem_index(tex)

    out: List[Dict] = []
    idx = 1
    seen_source_idx: set[str] = set()

    # Subpart coverage checks by exercise id.
    by_ex: Dict[str, Dict[str, set[str]]] = {}
    for row in ordered:
        ex_num = (row.get("exercise") or "").strip()
        part_id = (row.get("part") or "").strip().lower()
        if not ex_num or not part_id:
            continue
        by_ex.setdefault(ex_num, {"problem": set(), "proof": set()})
        if (row.get("problem") or "").strip():
            by_ex[ex_num]["problem"].add(part_id)
        if (row.get("proof") or "").strip():
            by_ex[ex_num]["proof"].add(part_id)

    for ex_num, mm in by_ex.items():
        pset = mm["problem"]
        fset = mm["proof"]
        miss_f = sorted(pset - fset)
        miss_p = sorted(fset - pset)
        if miss_f or miss_p:
            warnings.append(
                {
                    "type": "subpart_mismatch",
                    "exercise": ex_num,
                    "statement_parts": sorted(pset),
                    "proof_parts": sorted(fset),
                }
            )

    for row in ordered:
        ex_num = (row.get("exercise") or "").strip()
        part_id = (row.get("part") or "").strip().lower()
        problem = (row.get("problem") or "").strip()
        proof = (row.get("proof") or "").strip()
        raw_problem_tex = (row.get("raw_problem_tex") or "").strip()
        raw_proof_tex = (row.get("raw_proof_tex") or "").strip()
        raw_joined = (raw_problem_tex + "\n\n" + raw_proof_tex).strip()
        if ex_num and part_id:
            stem = (exercise_common_problem_prefix.get(ex_num) or "").strip()
            if stem:
                problem = _compose_with_common_prefix(stem, problem)

        if not ex_num:
            warnings.append({"type": "unknown_exercise", "part": part_id or None})
        # If this is an exercise-level item (no part) and the exercise has subpart proofs,
        # do not emit a false missing_proof warning for the parent slot.
        has_subpart_proof = bool(ex_num and not part_id and by_ex.get(ex_num, {}).get("proof"))
        if not proof and not (not part_id and has_subpart_proof):
            warnings.append({"type": "missing_proof", "source_idx": _source_idx(ex_num, part_id)})
        if part_id and not (len(part_id) == 1 and part_id.isalpha()):
            warnings.append({"type": "nonstandard_part_id", "exercise": ex_num, "part": part_id})

        ptype = infer_problem_type(problem)
        ptype = refine_problem_type(problem, proof, ptype, enable=enable_type_refine)
        d_ans = extract_direct_answer(ptype, proof) if enable_direct_answer else ""
        src_idx = _source_idx(ex_num, part_id)
        if src_idx in seen_source_idx:
            warnings.append({"type": "duplicate_source_idx", "source_idx": src_idx})
        seen_source_idx.add(src_idx)

        defined_tags = extract_defined_tags(raw_joined)
        body_refs, body_ref_evidence, unresolved_body_refs = extract_body_refs(raw_joined, all_tag_index)
        body_ref_targets = [dict(all_tag_index[t]) for t in body_refs if t in all_tag_index]

        theorem_refs = _merge_ordered_unique([], list(row.get("theorem_refs") or []))
        theorem_ref_evidence = dict(row.get("theorem_ref_evidence") or {})
        theorem_ref_targets = [dict(doc_theorem_index[t]) for t in theorem_refs if t in doc_theorem_index]

        problem_with_context = compose_problem_with_context(problem, body_ref_targets, theorem_ref_targets)

        dependency: Dict[str, Union[List, Dict]] = {
            "defined_tags": defined_tags,
            "defined_tag_labels": [f"eq:{t}" for t in defined_tags],
            "body_refs": body_refs,
            "body_ref_labels": [all_tag_index[t]["label_key"] for t in body_refs if t in all_tag_index],
            "body_ref_targets": body_ref_targets,
            "body_ref_evidence": body_ref_evidence,
            "unresolved_body_refs": unresolved_body_refs,
            "theorem_refs": theorem_refs,
            "theorem_ref_labels": [_theorem_label_key_from_ref(t) for t in theorem_refs],
            "theorem_ref_targets": theorem_ref_targets,
            "theorem_ref_evidence": theorem_ref_evidence,
        }

        # index must be contiguous starting from 1.
        item: Dict[str, Union[int, str, List, Dict]] = {
            "index": idx,
            "problem": problem,
            "proof": proof,
            "direct_answer": d_ans,
            "题目类型": [ptype],
            "预估难度": [],
            "source": source_name,
            "source_idx": src_idx,
            "defined_tags": defined_tags,
            "defined_tag_labels": [f"eq:{t}" for t in defined_tags],
            "body_refs": body_refs,
            "body_ref_labels": [all_tag_index[t]["label_key"] for t in body_refs if t in all_tag_index],
            "body_ref_targets": body_ref_targets,
            "body_ref_evidence": body_ref_evidence,
            "unresolved_body_refs": unresolved_body_refs,
            "theorem_refs": theorem_refs,
            "theorem_ref_labels": [_theorem_label_key_from_ref(t) for t in theorem_refs],
            "theorem_ref_targets": theorem_ref_targets,
            "theorem_ref_evidence": theorem_ref_evidence,
            "problem_with_context": problem_with_context,
            "dependency": dependency,
        }
        if include_raw:
            item["raw_problem_tex"] = raw_problem_tex
            item["raw_proof_tex"] = raw_proof_tex
        out.append(item)
        idx += 1

    return out, warnings


def _parse_source_idx(source_idx: str) -> Tuple[str, str]:
    m = SOURCE_IDX_RE.match(source_idx or "")
    if not m:
        return "", ""
    ex = (m.group("ex") or "").strip()
    part = (m.group("part") or "").strip().lower()
    return ex, part


def _reindex_rows(rows: List[Dict]) -> None:
    for i, row in enumerate(rows, start=1):
        row["index"] = i


def _compact_dependency_fields(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dep_keys = [
        "defined_tags",
        "defined_tag_labels",
        "body_refs",
        "body_ref_labels",
        "body_ref_targets",
        "body_ref_evidence",
        "unresolved_body_refs",
        "theorem_refs",
        "theorem_ref_labels",
        "theorem_ref_targets",
        "theorem_ref_evidence",
    ]

    keep_top = {
        "index",
        "problem",
        "proof",
        "direct_answer",
        "题目类型",
        "预估难度",
        "source",
        "source_idx",
        "problem_with_context",
        "dependency",
    }

    out: List[Dict[str, Any]] = []
    for row in rows:
        r = dict(row or {})
        dep = r.get("dependency")
        dep_obj: Dict[str, Any] = dict(dep) if isinstance(dep, dict) else {}

        for k in dep_keys:
            if k in r and k not in dep_obj:
                dep_obj[k] = r.get(k)
            elif k not in dep_obj:
                dep_obj[k] = [] if k.endswith("s") or k.endswith("_labels") or k.endswith("_targets") else {}

        r["dependency"] = dep_obj

        # Drop any non-core top-level keys except raw_* (kept only if explicitly requested by caller earlier).
        for k in list(r.keys()):
            if k in keep_top:
                continue
            if k.startswith("raw_"):
                continue
            r.pop(k, None)

        # Always remove duplicated dependency keys from top-level.
        for k in dep_keys:
            r.pop(k, None)

        out.append(r)
    return out


def validate_rows(rows: List[Dict]) -> List[Dict]:
    """
    Validate final rows and emit machine-readable warnings.
    This validator drives iterative self-repair.
    """
    warnings: List[Dict] = []
    seen_source_idx: Set[str] = set()
    by_ex: Dict[str, Dict[str, Set[str]]] = {}
    proof_parts_by_ex: Dict[str, Set[str]] = {}

    # Pre-scan: collect exercises that already have subpart proofs.
    for row in rows:
        sid0 = str(row.get("source_idx") or "").strip()
        proof0 = str(row.get("proof") or "").strip()
        ex0, part0 = _parse_source_idx(sid0)
        if ex0 and ex0.upper() != "UNKNOWN" and part0 and proof0:
            proof_parts_by_ex.setdefault(ex0, set()).add(part0)

    for row in rows:
        sid = str(row.get("source_idx") or "").strip()
        problem = str(row.get("problem") or "").strip()
        proof = str(row.get("proof") or "").strip()
        ptype = row.get("题目类型")

        if not sid:
            warnings.append({"type": "missing_source_idx"})
            ex, part = "", ""
        else:
            if sid in seen_source_idx:
                warnings.append({"type": "duplicate_source_idx", "source_idx": sid})
            seen_source_idx.add(sid)
            ex, part = _parse_source_idx(sid)
            if not ex:
                warnings.append({"type": "malformed_source_idx", "source_idx": sid})
            elif ex.upper() == "UNKNOWN":
                warnings.append({"type": "unknown_exercise", "source_idx": sid})
            if part and not (len(part) == 1 and part.isalpha()):
                warnings.append({"type": "nonstandard_part_id", "source_idx": sid, "part": part})

        if not problem:
            warnings.append({"type": "missing_problem", "source_idx": sid})
        has_subpart_proof = bool(ex and not part and proof_parts_by_ex.get(ex))
        if not proof and not has_subpart_proof:
            warnings.append({"type": "missing_proof", "source_idx": sid})
        if not isinstance(ptype, list) or len(ptype) != 1 or ptype[0] not in {"证明题", "求值题", "其他"}:
            warnings.append({"type": "invalid_problem_type", "source_idx": sid, "题目类型": ptype})

        if ex and ex.upper() != "UNKNOWN" and part:
            by_ex.setdefault(ex, {"problem": set(), "proof": set()})
            if problem:
                by_ex[ex]["problem"].add(part)
            if proof:
                by_ex[ex]["proof"].add(part)

    for ex, mm in by_ex.items():
        pset = mm["problem"]
        fset = mm["proof"]
        if pset != fset:
            warnings.append(
                {
                    "type": "subpart_mismatch",
                    "exercise": ex,
                    "statement_parts": sorted(pset),
                    "proof_parts": sorted(fset),
                }
            )
    return warnings


def _safe_json_load(s: str) -> Optional[Dict]:
    try:
        v = json.loads(s)
    except Exception:
        return None
    return v if isinstance(v, dict) else None


def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Best-effort extraction of Retry-After from OpenAI-compatible error responses."""
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    headers = getattr(resp, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(1.0, min(20.0, float(raw)))
    except Exception:
        return None


def llm_call(client: object, model: str, prompt: str, *, max_tokens: int) -> str:
    """
    Call chat model with retry/backoff and return text content.
    Handles gateways that require stream=true.
    """

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

    attempts = 8
    for k in range(1, attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
            )
            time.sleep(1.0)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if _is_stream_required_error(e):
                stream_obj = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=max_tokens,
                    stream=True,
                )
                time.sleep(1.0)
                return (_collect_stream_text(stream_obj) or "").strip()

            name = e.__class__.__name__
            retryable = name in _RETRYABLE_OPENAI_ERRORS
            if (not retryable) or k >= attempts:
                raise
            ra = _retry_after_seconds(e)
            backoff = min(20.0, max(5.0, float(2 ** (k - 1))))
            time.sleep(ra if ra is not None else backoff)
    return ""


def _default_cache_dir() -> Path:
    """
    Small on-disk cache to reduce token usage across re-runs.
    """
    d = Path.cwd() / ".cache_texTojson"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        d = Path("/tmp") / ".cache_texTojson"
        d.mkdir(parents=True, exist_ok=True)
    return d


def llm_call_cached(
    client: object,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    cache_dir: Optional[Path] = None,
    cache_enabled: bool = True,
) -> str:
    """
    Cache raw LLM responses keyed by (model, max_tokens, prompt).
    """
    if not cache_enabled:
        return llm_call(client, model, prompt, max_tokens=max_tokens)

    cache_dir = cache_dir or _default_cache_dir()
    key = hashlib.sha256((model + "\n" + str(max_tokens) + "\n" + prompt).encode("utf-8")).hexdigest()
    path = cache_dir / f"{key}.txt"

    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            pass

    out = llm_call(client, model, prompt, max_tokens=max_tokens)
    try:
        path.write_text(out, encoding="utf-8")
    except Exception:
        pass
    return out


def _extract_first_json_object(text: str) -> Optional[Dict]:
    s = (text or "").strip()
    if not s:
        return None
    direct = _safe_json_load(s)
    if direct is not None:
        return direct
    l = s.find("{")
    r = s.rfind("}")
    if l >= 0 and r > l:
        return _safe_json_load(s[l : r + 1])
    return None


def _llm_self_check_single(
    *,
    model: str,
    api_key: str,
    source_idx: str,
    problem: str,
    proof: str,
    problem_with_context: str,
    body_refs: List[str],
    body_ref_targets: List[Dict[str, Any]],
    ptype_list: List[str],
    direct_answer: str,
    warnings_for_item: List[Dict],
    cache_dir: Optional[Path] = None,
    cache_enabled: bool = True,
) -> Optional[Dict[str, Union[str, List[str]]]]:
    """
    Optional LLM-assisted validator/repairer.
    Returns None if unavailable, API error, or invalid output format.
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None

    client = OpenAI(api_key=api_key)
    sys_prompt = (
        "You are a strict JSON checker for exercise records. "
        "Minimally repair obvious issues while preserving original wording. "
        "Do not invent facts. "
        "Use problem_with_context and body_ref_targets as supporting context when present. "
        "When body_ref_targets include equation_content, treat equation_content as authoritative context "
        "and ignore surrounding narrative snippets. "
        "For multipart exercises with multiple solutions, keep per-part alignment stable: "
        "do not move proof content across source_idx parts. "
        "If source_idx is Exercise x.y-(p), ensure problem keeps the exercise stem/preamble "
        "that appears before the first (a) when present. "
        "Output ONLY one JSON object with keys: "
        "problem, proof, direct_answer, 题目类型, notes. "
        "题目类型 must be a one-element list from {证明题, 求值题, 其他}."
    )
    payload = {
        "source_idx": source_idx,
        "problem": problem,
        "proof": proof,
        "problem_with_context": problem_with_context,
        "body_refs": body_refs,
        "body_ref_targets": body_ref_targets,
        "direct_answer": direct_answer,
        "题目类型": ptype_list,
        "warnings": warnings_for_item,
        "rules": [
            "Keep wording/order as much as possible.",
            "If proof is missing, you may provide a concise logically consistent proof based on the problem.",
            "If problem is empty, do not fabricate content.",
            "Do not remove shared preamble/stem from multipart problems.",
            "Do not mix proofs between different parts (a)/(b)/(c).",
            "Do not edit fields not requested in output schema.",
            "题目类型 must be one of 证明题/求值题/其他.",
        ],
    }
    prompt = (
        sys_prompt
        + "\n\nInput JSON:\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n\nOutput JSON only."
    )
    try:
        raw = llm_call_cached(
            client,
            model,
            prompt,
            max_tokens=1400,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
        )
    except Exception:
        return None

    obj = _extract_first_json_object(raw)
    if not obj:
        return None

    out_problem = str(obj.get("problem") or "").strip()
    out_proof = str(obj.get("proof") or "").strip()
    out_da = str(obj.get("direct_answer") or "").strip()
    out_type = obj.get("题目类型")
    if not isinstance(out_type, list) or not out_type:
        return None
    t0 = str(out_type[0]).strip()
    if t0 not in {"证明题", "求值题", "其他"}:
        return None
    return {
        "problem": out_problem,
        "proof": out_proof,
        "direct_answer": out_da,
        "题目类型": [t0],
    }


def _llm_extract_direct_answer_single(
    *,
    model: str,
    api_key: str,
    source_idx: str,
    problem: str,
    proof: str,
    problem_with_context: str,
    body_ref_targets: List[Dict[str, Any]],
    cache_dir: Optional[Path] = None,
    cache_enabled: bool = True,
) -> str:
    """
    Optional LLM fallback extractor for direct answers.
    Returns a concise answer string or "".
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return ""

    client = OpenAI(api_key=api_key)
    prompt = (
        "You are extracting direct answers from math exercise solutions.\n"
        "Return ONLY the final direct answer text (no explanations, no markdown).\n"
        "If no explicit direct/numeric/closed-form answer can be identified, return empty string.\n"
        "Keep LaTeX as-is if needed.\n\n"
        "Never rewrite problem/proof text and never infer answers from other parts.\n"
        "Use problem_with_context/body_ref_targets only as disambiguation context.\n"
        "When body_ref_targets include equation_content, use equation_content and ignore narrative context.\n"
        f"source_idx: {source_idx}\n\n"
        "Problem with context:\n"
        f"{problem_with_context}\n\n"
        "Body reference targets:\n"
        f"{json.dumps(body_ref_targets, ensure_ascii=False)}\n\n"
        "Problem:\n"
        f"{problem}\n\n"
        "Proof/Solution:\n"
        f"{proof}\n"
    )
    try:
        raw = llm_call_cached(
            client,
            model,
            prompt,
            max_tokens=300,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
        )
    except Exception:
        return ""
    out = (raw or "").strip()
    out = re.sub(r"^```(?:text|latex|json)?\s*", "", out, flags=re.IGNORECASE).strip()
    out = re.sub(r"\s*```$", "", out).strip()
    return out


def _llm_verify_problem_type_single(
    *,
    model: str,
    api_key: str,
    source_idx: str,
    problem: str,
    proof: str,
    problem_with_context: str,
    body_refs: List[str],
    body_ref_targets: List[Dict[str, Any]],
    current_type: str,
    cache_dir: Optional[Path] = None,
    cache_enabled: bool = True,
) -> Optional[str]:
    """
    Optional one-shot LLM verifier for problem type.
    Returns one of {证明题, 求值题, 其他} or None on failure.
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None

    client = OpenAI(api_key=api_key)
    payload = {
        "source_idx": source_idx,
        "problem": problem,
        "proof": proof,
        "problem_with_context": problem_with_context,
        "body_refs": body_refs,
        "body_ref_targets": body_ref_targets,
        "current_type": current_type,
        "allowed": ["证明题", "求值题", "其他"],
    }
    prompt = (
        "You are validating math exercise type labels. "
        "Choose exactly one type from {证明题, 求值题, 其他}. "
        "Prefer semantic intent of the problem statement; use proof only as weak evidence. "
        "Use problem_with_context/body_ref_targets when they help resolve omitted references. "
        "When body_ref_targets include equation_content, prioritize equation_content over narrative snippets. "
        "Output ONLY one JSON object with key 题目类型 and a single value string.\n\n"
        "Input JSON:\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    try:
        raw = llm_call_cached(
            client,
            model,
            prompt,
            max_tokens=180,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
        )
    except Exception:
        return None

    obj = _extract_first_json_object(raw)
    if isinstance(obj, dict):
        t = str(obj.get("题目类型") or "").strip()
        if t in {"证明题", "求值题", "其他"}:
            return t

    raw_t = (raw or "").strip()
    if raw_t in {"证明题", "求值题", "其他"}:
        return raw_t
    return None


def apply_llm_type_verification(
    rows: List[Dict],
    *,
    enable: bool,
    model: str,
    max_items: int,
    cache_dir: Optional[Path] = None,
    cache_enabled: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Optional post-pass for hybrid classification:
    - keep deterministic rule output as baseline
    - let LLM verify (and optionally adjust) type once
    """
    if not enable:
        return rows, []

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return rows, [{"type": "llm_type_check_skipped", "reason": "missing_openai_api_key"}]

    out_rows: List[Dict] = []
    touched = 0
    changed = 0
    budget = max(0, int(max_items))

    for row in rows:
        current_list = row.get("题目类型")
        current_type = "其他"
        if isinstance(current_list, list) and len(current_list) == 1 and isinstance(current_list[0], str):
            current_type = current_list[0]

        problem_text = str(row.get("problem") or "")
        value_hit, proof_hit = _problem_type_signal_flags(problem_text)

        # Route to LLM when:
        # 1) conflict samples (both value/proof signals hit), or
        # 2) deterministic label is "其他".
        need_llm = (value_hit and proof_hit) or (current_type == "其他")
        if not need_llm:
            out_rows.append(row)
            continue
        if touched >= budget:
            out_rows.append(row)
            continue

        touched += 1
        new_t = _llm_verify_problem_type_single(
            model=model,
            api_key=api_key,
            source_idx=str(row.get("source_idx") or ""),
            problem=problem_text,
            proof=str(row.get("proof") or ""),
            problem_with_context=str(row.get("problem_with_context") or ""),
            body_refs=list(row.get("body_refs") or []),
            body_ref_targets=list(row.get("body_ref_targets") or []),
            current_type=current_type,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
        )
        if not new_t:
            out_rows.append(row)
            continue

        if new_t != current_type:
            upd = dict(row)
            upd["题目类型"] = [new_t]
            out_rows.append(upd)
            changed += 1
        else:
            out_rows.append(row)

    meta = [
        {
            "type": "llm_type_check_summary",
            "touched_items": touched,
            "changed_items": changed,
            "model": model,
        }
    ]
    return out_rows, meta


def apply_llm_direct_answer_fallback(
    rows: List[Dict],
    *,
    enable: bool,
    model: str,
    max_items: int,
    cache_dir: Optional[Path] = None,
    cache_enabled: bool = True,
) -> List[Dict]:
    """
    Optional post-pass for direct_answer:
    - only for 求值题
    - use LLM extraction for every matched row
    - if first LLM extraction is empty, retry once
    - if still empty, re-judge 题目类型
    - if LLM returns empty and type unchanged, keep existing direct_answer
    - max_items <= 0 means no limit
    """
    if not enable:
        return rows
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return rows

    out_rows: List[Dict] = []
    touched = 0
    budget = int(max_items)
    unlimited = budget <= 0
    for row in rows:
        ptype = row.get("题目类型")
        current_da = str(row.get("direct_answer") or "").strip()
        if (
            (unlimited or touched < budget)
            and isinstance(ptype, list)
            and len(ptype) == 1
            and ptype[0] == "求值题"
        ):
            touched += 1
            new_da = _llm_extract_direct_answer_single(
                model=model,
                api_key=api_key,
                source_idx=str(row.get("source_idx") or ""),
                problem=str(row.get("problem") or ""),
                proof=str(row.get("proof") or ""),
                problem_with_context=str(row.get("problem_with_context") or ""),
                body_ref_targets=list(row.get("body_ref_targets") or []),
                cache_dir=cache_dir,
                cache_enabled=cache_enabled,
            )

            # Retry once when first extraction is empty.
            if not new_da:
                new_da = _llm_extract_direct_answer_single(
                    model=model,
                    api_key=api_key,
                    source_idx=str(row.get("source_idx") or ""),
                    problem=str(row.get("problem") or ""),
                    proof=str(row.get("proof") or ""),
                    problem_with_context=str(row.get("problem_with_context") or ""),
                    body_ref_targets=list(row.get("body_ref_targets") or []),
                    cache_dir=cache_dir,
                    cache_enabled=cache_enabled,
                )

            if new_da:
                upd = dict(row)
                upd["direct_answer"] = "\\[\n" + new_da + "\n\\]"
                out_rows.append(upd)
                continue

            # Still empty after retry: re-judge problem type.
            problem_text = str(row.get("problem") or "")
            proof_text = str(row.get("proof") or "")
            sid = str(row.get("source_idx") or "")
            retyped = _llm_verify_problem_type_single(
                model=model,
                api_key=api_key,
                source_idx=sid,
                problem=problem_text,
                proof=proof_text,
                problem_with_context=str(row.get("problem_with_context") or ""),
                body_refs=list(row.get("body_refs") or []),
                body_ref_targets=list(row.get("body_ref_targets") or []),
                current_type="求值题",
                cache_dir=cache_dir,
                cache_enabled=cache_enabled,
            )
            if not retyped:
                retyped = infer_problem_type(problem_text)

            if retyped and retyped != "求值题":
                upd = dict(row)
                upd["题目类型"] = [retyped]
                # Re-typed away from value question: clear stale direct_answer.
                upd["direct_answer"] = ""
                out_rows.append(upd)
                continue

            # keep original direct_answer when type unchanged and extraction failed
            if current_da:
                out_rows.append(row)
                continue
        out_rows.append(row)
    return out_rows


def apply_llm_self_check(
    rows: List[Dict],
    warnings: List[Dict],
    *,
    enable: bool,
    model: str,
    max_items: int,
    cache_dir: Optional[Path] = None,
    cache_enabled: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Optional post-pass:
    - Default disabled; deterministic baseline remains unchanged.
    - Targets rows tied to warning signals (e.g., missing_proof).
    """
    if not enable:
        return rows, warnings

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return rows, list(warnings) + [{"type": "llm_self_check_skipped", "reason": "missing_openai_api_key"}]

    warn_map: Dict[str, List[Dict]] = {}
    for w in warnings:
        if not isinstance(w, dict):
            continue
        sid = str(w.get("source_idx") or "").strip()
        if sid:
            warn_map.setdefault(sid, []).append(w)

    # Exercise-level warning fan-out (subpart mismatch) -> all rows of that exercise.
    for w in warnings:
        if not isinstance(w, dict) or w.get("type") != "subpart_mismatch":
            continue
        ex = str(w.get("exercise") or "").strip()
        if not ex:
            continue
        prefix = f"Exercise {ex}"
        for row in rows:
            sid = str(row.get("source_idx") or "")
            if sid.startswith(prefix):
                warn_map.setdefault(sid, []).append(w)

    touched = 0
    patched = 0
    out_rows: List[Dict] = []

    for row in rows:
        sid = str(row.get("source_idx") or "")
        row_warns = warn_map.get(sid, [])
        if not row_warns or touched >= max_items:
            out_rows.append(row)
            continue
        touched += 1

        fixed = _llm_self_check_single(
            model=model,
            api_key=api_key,
            source_idx=sid,
            problem=str(row.get("problem") or ""),
            proof=str(row.get("proof") or ""),
            problem_with_context=str(row.get("problem_with_context") or ""),
            body_refs=list(row.get("body_refs") or []),
            body_ref_targets=list(row.get("body_ref_targets") or []),
            ptype_list=list(row.get("题目类型") or ["其他"]),
            direct_answer=str(row.get("direct_answer") or ""),
            warnings_for_item=row_warns,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
        )
        if not fixed:
            out_rows.append(row)
            continue

        # Minimal patch strategy: non-empty problem/proof overwrite, plus type/direct_answer.
        upd = dict(row)
        if fixed["problem"]:
            upd["problem"] = fixed["problem"]
        if fixed["proof"]:
            upd["proof"] = fixed["proof"]
        upd["题目类型"] = fixed["题目类型"]
        upd["direct_answer"] = fixed["direct_answer"]
        out_rows.append(upd)
        patched += 1

    out_warnings = list(warnings) + [
        {
            "type": "llm_self_check_summary",
            "touched_items": touched,
            "patched_items": patched,
            "model": model,
        }
    ]
    return out_rows, out_warnings


def run_iterative_self_repair(
    rows: List[Dict],
    *,
    enable: bool,
    model: str,
    max_items: int,
    max_rounds: int,
    require_clean: bool,
    cache_dir: Optional[Path] = None,
    cache_enabled: bool = True,
    llm_direct_answer_fallback: bool = False,
    llm_direct_answer_max_items: int = 0,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Iteratively run validator + optional LLM patching until:
    - no warnings, or
    - max rounds reached, or
    - no progress.
    """
    _reindex_rows(rows)
    warnings = validate_rows(rows)
    if not enable:
        if require_clean and warnings:
            raise RuntimeError(f"Validation failed without self-repair: {len(warnings)} warnings")
        return rows, warnings
    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        raise RuntimeError("LLM self-repair enabled but OPENAI_API_KEY is not set.")

    prev_signature = ""
    for _ in range(max(1, max_rounds)):
        if not warnings:
            break
        signature = json.dumps(warnings, ensure_ascii=False, sort_keys=True)
        if signature == prev_signature:
            break
        prev_signature = signature

        rows, _meta = apply_llm_self_check(
            rows,
            warnings,
            enable=True,
            model=model,
            max_items=max_items,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
        )
        _reindex_rows(rows)
        warnings = validate_rows(rows)

    if require_clean and warnings:
        raise RuntimeError(
            f"Self-repair finished but warnings remain ({len(warnings)}). "
            "You can inspect --warnings-json."
        )
    rows = apply_llm_direct_answer_fallback(
        rows,
        enable=llm_direct_answer_fallback,
        model=model,
        max_items=max(0, int(llm_direct_answer_max_items)),
        cache_dir=cache_dir,
        cache_enabled=cache_enabled,
    )
    return rows, warnings




def infer_md_path_from_tex(in_tex: Path, in_md_arg: str = "") -> Optional[Path]:
    if in_md_arg:
        p = Path(in_md_arg).expanduser().resolve()
        if p.exists():
            return p

    # Fixed pipeline path: use the first-stage pdfTomd output only.
    cand = in_tex.with_suffix(".md")
    if cand.exists():
        return cand

    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert current book TeX blocks to JSON records.")
    ap.add_argument("in_tex", type=str, help="Input .tex file")
    ap.add_argument("out_json", type=str, help="Output .json file")
    ap.add_argument(
        "--source-name",
        type=str,
        default="",
        help="Override source field (defaults to SOURCE_NAME_ENTRY if omitted)",
    )
    ap.add_argument("--warnings-json", type=str, default="", help="Optional warnings output .json")
    ap.add_argument("--emit-raw", action="store_true", help="Include raw_problem_tex/raw_proof_tex in output")
    ap.add_argument("--type-refine", action="store_true", help="Enable optional second-stage rule refine")
    ap.add_argument("--no-direct-answer", action="store_true", help="Disable direct_answer extraction")
    ap.add_argument("--llm-self-check", action="store_true", help="Enable optional LLM self-check repair pass")
    ap.add_argument("--llm-model", type=str, default="gpt-5-mini", help="LLM model for self-check pass")
    ap.add_argument("--llm-max-items", type=int, default=20, help="Max warned rows sent to LLM self-check")
    ap.add_argument("--llm-max-rounds", type=int, default=3, help="Max iterative self-repair rounds")
    ap.add_argument("--llm-cache-dir", type=str, default="", help="Optional cache directory for LLM calls")
    ap.add_argument("--llm-no-cache", action="store_true", help="Disable on-disk LLM cache")
    ap.add_argument(
        "--llm-type-check",
        action="store_true",
        help="Run one LLM verification pass for 题目类型 after rule-based classification",
    )
    ap.add_argument(
        "--llm-type-max-items",
        type=int,
        default=200,
        help="Max rows sent to LLM type verification pass",
    )
    ap.add_argument(
        "--llm-direct-answer-fallback",
        action="store_true",
        help="Use LLM extractor for direct_answer on 求值题 rows",
    )
    ap.add_argument(
        "--llm-direct-answer-max-items",
        type=int,
        default=0,
        help="Max rows for LLM direct_answer extraction (<=0 means all 求值题)",
    )
    ap.add_argument("--in-md", type=str, default="", help="Optional input .md used as body-ref fallback")
    ap.add_argument(
        "--require-clean",
        action="store_true",
        help="Fail if warnings remain after iterative self-repair",
    )
    ap.add_argument(
        "--final-json-only",
        action="store_true",
        help="Strict final mode: LLM self-check + require clean + print only final JSON to stdout",
    )
    args = ap.parse_args()

    if args.final_json_only:
        args.llm_self_check = True
        args.require_clean = True
        args.warnings_json = ""

    in_tex = Path(args.in_tex).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    if not in_tex.exists():
        raise FileNotFoundError(f"Input tex not found: {in_tex}")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    tex = in_tex.read_text(encoding="utf-8")
    md_path = infer_md_path_from_tex(in_tex, str(args.in_md or ""))
    md_text = md_path.read_text(encoding="utf-8") if (md_path and md_path.exists()) else ""

    rows, warns = build_records_from_tex(
        tex,
        source_name=(args.source_name if args.source_name else SOURCE_NAME_ENTRY),
        include_raw=bool(args.emit_raw),
        enable_type_refine=bool(args.type_refine),
        enable_direct_answer=not args.no_direct_answer,
        md_text=md_text,
    )
    rows, type_meta = apply_llm_type_verification(
        rows,
        enable=bool(args.llm_type_check),
        model=str(args.llm_model),
        max_items=max(0, int(args.llm_type_max_items)),
        cache_dir=(Path(args.llm_cache_dir).expanduser().resolve() if args.llm_cache_dir else None),
        cache_enabled=not bool(args.llm_no_cache),
    )
    rows, warns = run_iterative_self_repair(
        rows,
        enable=bool(args.llm_self_check),
        model=str(args.llm_model),
        max_items=max(0, int(args.llm_max_items)),
        max_rounds=max(1, int(args.llm_max_rounds)),
        require_clean=bool(args.require_clean),
        cache_dir=(Path(args.llm_cache_dir).expanduser().resolve() if args.llm_cache_dir else None),
        cache_enabled=not bool(args.llm_no_cache),
        llm_direct_answer_fallback=bool(args.llm_direct_answer_fallback),
        llm_direct_answer_max_items=max(0, int(args.llm_direct_answer_max_items)),
    )
    if type_meta:
        warns = list(warns) + list(type_meta)

    rows = _compact_dependency_fields(rows)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.warnings_json and not args.final_json_only:
        wp = Path(args.warnings_json).expanduser().resolve()
        wp.parent.mkdir(parents=True, exist_ok=True)
        wp.write_text(json.dumps(warns, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"WARNINGS: {wp} (count={len(warns)})")

    if args.final_json_only:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(f"DONE: {out_json} (items={len(rows)})")


if __name__ == "__main__":
    main()
