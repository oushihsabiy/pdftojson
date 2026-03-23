#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""General-purpose BLOCK-sentinel merger for TeX files.

Default behavior (backward compatible):
- Extract whole chunks delimited by:
    %<BLOCK ...>
    ...
    %</BLOCK>
- Append extracted chunks to target, right before \end{document} when present.
- De-duplicate by normalized full block text against target blocks.

Extra capabilities:
- Filter by block type: --include-types / --exclude-types
- Choose insertion position: --position before-end-document|end
- Dry run: --dry-run (print stats only, do not write)
- Optional marker suppression: --no-marker
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BLOCK_START_LINE_RE = re.compile(r"^\s*%<BLOCK\s+([^>]+)>\s*$")
BLOCK_END_LINE_RE = re.compile(r"^\s*%</BLOCK>\s*$")
KV_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=("(?:\\.|[^"])*"|\S+)')
END_DOC_RE = re.compile(r"\\end\{document\}", re.IGNORECASE)


@dataclass
class BlockChunk:
    raw: str
    attrs: Dict[str, str]

    @property
    def btype(self) -> str:
        return (self.attrs.get("type") or "").strip().lower()


def _unescape_attr(s: str) -> str:
    t = s
    if t.startswith('"') and t.endswith('"'):
        t = t[1:-1]
    t = t.replace("\\n", "\n")
    t = t.replace('\\"', '"')
    t = t.replace("\\\\", "\\")
    return t


def _parse_attrs(header_body: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in KV_RE.finditer(header_body or ""):
        out[m.group(1)] = _unescape_attr(m.group(2))
    return out


def extract_blocks(tex: str) -> List[BlockChunk]:
    """Parse top-level sentinel blocks with line-based scan (robust to spacing)."""
    lines = (tex or "").splitlines()
    out: List[BlockChunk] = []

    i = 0
    while i < len(lines):
        m = BLOCK_START_LINE_RE.match(lines[i] or "")
        if not m:
            i += 1
            continue

        start = i
        depth = 1
        i += 1
        while i < len(lines):
            ln = lines[i] or ""
            if BLOCK_START_LINE_RE.match(ln):
                depth += 1
            elif BLOCK_END_LINE_RE.match(ln):
                depth -= 1
                if depth == 0:
                    break
            i += 1

        if i >= len(lines):
            # Unclosed block: consume till EOF conservatively.
            raw = "\n".join(lines[start:]).rstrip()
        else:
            raw = "\n".join(lines[start : i + 1]).rstrip()

        attrs = _parse_attrs(m.group(1) or "")
        out.append(BlockChunk(raw=raw, attrs=attrs))
        i += 1

    return out


def normalize_block_text(block: str) -> str:
    lines = [ln.rstrip() for ln in (block or "").splitlines()]
    return "\n".join(lines).strip()


def ensure_origintext_env(target_tex: str, incoming_blocks: List[BlockChunk]) -> str:
    uses_origin = any("\\begin{origintext}" in (b.raw or "") for b in (incoming_blocks or []))
    if not uses_origin:
        return target_tex
    if "\\newenvironment{origintext}{}{}" in (target_tex or ""):
        return target_tex

    m = re.search(r"\\begin\{document\}", target_tex or "", flags=re.IGNORECASE)
    decl = "\\newenvironment{origintext}{}{}\n"
    if m:
        head = (target_tex or "")[: m.start()].rstrip()
        tail = (target_tex or "")[m.start() :]
        return head + "\n" + decl + "\n" + tail
    return decl + "\n" + (target_tex or "")


def insert_payload(target_tex: str, payload: str, *, position: str) -> str:
    if not payload.strip():
        return target_tex

    if position == "end":
        base = (target_tex or "").rstrip()
        return base + "\n\n" + payload.rstrip() + "\n"

    # default: before-end-document
    matches = list(END_DOC_RE.finditer(target_tex or ""))
    if not matches:
        base = (target_tex or "").rstrip()
        return base + "\n\n" + payload.rstrip() + "\n"

    m = matches[-1]
    head = (target_tex or "")[: m.start()].rstrip()
    tail = (target_tex or "")[m.start() :]
    return head + "\n\n" + payload.rstrip() + "\n\n" + tail


def _split_csv(s: str) -> List[str]:
    vals: List[str] = []
    for x in (s or "").split(","):
        t = x.strip().lower()
        if t:
            vals.append(t)
    return vals


def _want_block(b: BlockChunk, include_types: List[str], exclude_types: List[str]) -> bool:
    bt = b.btype
    if include_types and bt not in include_types:
        return False
    if exclude_types and bt in exclude_types:
        return False
    return True


def merge_blocks(
    source_text: str,
    target_text: str,
    *,
    dedupe: bool = True,
    include_types: Optional[List[str]] = None,
    exclude_types: Optional[List[str]] = None,
    position: str = "before-end-document",
    add_marker: bool = True,
) -> Tuple[str, int, int, int]:
    include_types = include_types or []
    exclude_types = exclude_types or []

    src_all = extract_blocks(source_text)
    tgt_all = extract_blocks(target_text)

    src_blocks = [b for b in src_all if _want_block(b, include_types, exclude_types)]

    kept: List[BlockChunk] = []
    skipped = 0

    if dedupe:
        seen = {normalize_block_text(b.raw) for b in tgt_all}
        for b in src_blocks:
            nb = normalize_block_text(b.raw)
            if nb in seen:
                skipped += 1
                continue
            seen.add(nb)
            kept.append(b)
    else:
        kept = src_blocks

    if not kept:
        return target_text, len(src_blocks), 0, skipped

    target_text = ensure_origintext_env(target_text, kept)

    chunks = [b.raw for b in kept]
    marker = ""
    if add_marker:
        marker = (
            "% ===== merged BLOCK chunks =====\n"
            "% source blocks appended by merge_tex_blocks.py\n\n"
        )
    payload = marker + "\n\n".join(chunks)

    merged = insert_payload(target_text, payload, position=position)
    return merged, len(src_blocks), len(kept), skipped


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge %<BLOCK ...>%</BLOCK> chunks from one TeX file into another")
    ap.add_argument("--from-tex", dest="from_tex", required=True, help="Source TeX (provide blocks)")
    ap.add_argument("--to-tex", dest="to_tex", required=True, help="Target TeX (receive blocks)")
    ap.add_argument("--out-tex", dest="out_tex", default="", help="Output TeX (default: overwrite --to-tex)")

    ap.add_argument("--no-dedupe", action="store_true", help="Do not de-duplicate blocks")
    ap.add_argument(
        "--include-types",
        default="",
        help="Comma-separated block types to include only (e.g. origintext,thm)",
    )
    ap.add_argument(
        "--exclude-types",
        default="",
        help="Comma-separated block types to exclude (e.g. proof)",
    )
    ap.add_argument(
        "--position",
        choices=["before-end-document", "end"],
        default="before-end-document",
        help="Where to insert merged blocks in target TeX",
    )
    ap.add_argument("--no-marker", action="store_true", help="Do not add merge marker comments")
    ap.add_argument("--dry-run", action="store_true", help="Show stats only; do not write output")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    src = Path(args.from_tex).expanduser().resolve()
    dst = Path(args.to_tex).expanduser().resolve()
    out = Path(args.out_tex).expanduser().resolve() if args.out_tex else dst

    if not src.exists():
        raise FileNotFoundError(f"Source tex not found: {src}")
    if not dst.exists():
        raise FileNotFoundError(f"Target tex not found: {dst}")

    src_text = src.read_text(encoding="utf-8")
    dst_text = dst.read_text(encoding="utf-8")

    include_types = _split_csv(args.include_types)
    exclude_types = _split_csv(args.exclude_types)

    merged, src_total, appended, skipped = merge_blocks(
        source_text=src_text,
        target_text=dst_text,
        dedupe=(not args.no_dedupe),
        include_types=include_types,
        exclude_types=exclude_types,
        position=str(args.position),
        add_marker=(not args.no_marker),
    )

    if not args.dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(merged, encoding="utf-8")

    print(
        f"DONE: {out} | source_blocks={src_total} appended={appended} "
        f"skipped={skipped} dedupe={not args.no_dedupe} "
        f"include_types={include_types or 'ALL'} exclude_types={exclude_types or 'NONE'} "
        f"position={args.position} dry_run={bool(args.dry_run)}"
    )


if __name__ == "__main__":
    main()
