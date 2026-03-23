#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pipeline runner with resume support for book and paper modes."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_PDF_DIR: Path
OUTPUT_JSON_DIR: Path
WORK_DIR: Path
PIPELINE_MODE: str
OCR_MAX_TOKENS: Optional[int]
ONLY_THESE_STEMS: set[str]
OVERWRITE_JSON: bool
OCR_WORKERS: Optional[int]
THINK_WORKERS: Optional[int]
STRICT_RESUME: bool
ATOMIC_OUTPUTS: bool
CLEAN_STALE_TMPS: bool
TEX_WARNINGS_JSON: bool
TEX_EMIT_RAW: bool
TEX_TYPE_REFINE: bool
TEX_NO_DIRECT_ANSWER: bool
TEX_LLM_SELF_CHECK: bool
TEX_LLM_MODEL: str
TEX_LLM_MAX_ITEMS: int
TEX_LLM_MAX_ROUNDS: int
TEX_LLM_CACHE_DIR: str
TEX_LLM_NO_CACHE: bool
TEX_LLM_TYPE_CHECK: bool
TEX_LLM_TYPE_MAX_ITEMS: int
TEX_LLM_DIRECT_ANSWER_FALLBACK: bool
TEX_LLM_DIRECT_ANSWER_MAX_ITEMS: int
TEX_REQUIRE_CLEAN: bool
TEX_FINAL_JSON_ONLY: bool
MDTOTEX_SKIP_TAG_RECOVERY: bool
BOOK_ENABLE_DOUBLE_ROUTE: bool
BOOK_DOUBLE_DIRNAME: str
BOOK_MERGE_NO_DEDUPE: bool
BOOK_MERGE_INCLUDE_TYPES: str
BOOK_MERGE_EXCLUDE_TYPES: str
BOOK_MERGE_POSITION: str
OUTPUT_JSON_NATURALIZED_DIR: Path
ENABLE_NATURALIZE: bool
NATURALIZE_MODEL: str
NATURALIZE_MAX_TOKENS: int
NATURALIZE_MAX_ITEMS: int
NATURALIZE_PROMPT_VERSION: str
NATURALIZE_CACHE_DIR: str
NATURALIZE_NO_CACHE: bool
NATURALIZE_DISABLE_LLM: bool
NATURALIZE_FORCE: bool
NATURALIZE_CLEAN_CACHE_ON_EXIT: bool


def find_settings_json() -> Path:
    path = PROJECT_ROOT / "settings.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing settings file: {path}")
    return path


def load_settings() -> dict[str, Any]:
    path = find_settings_json()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def get_setting(settings: dict[str, Any], key: str, default: Any) -> Any:
    return settings.get(key, default)


def run_cmd(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    env = os.environ.copy()
    # Some OpenAI/httpx setups fail on plain "socks://..." proxy URLs.
    # Keep compatible proxies, drop only invalid socks entries.
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        v = env.get(k, "")
        if isinstance(v, str) and v.lower().startswith("socks://"):
            env.pop(k, None)
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def ensure_scripts_exist(script_dir: Path, mode: str) -> tuple[Path, Path, Path]:
    if mode not in {"book", "paper"}:
        raise SystemExit(f"Unsupported mode: {mode!r} (expected 'book' or 'paper')")

    src_dir = script_dir / "src" / mode
    pdf_to_md = src_dir / "pdfTomd.py"
    md_to_tex = src_dir / "mdTotex.py"
    tex_to_json = src_dir / "texTojson.py"
    missing = [p for p in [pdf_to_md, md_to_tex, tex_to_json] if not p.exists()]
    if missing:
        raise SystemExit("Missing scripts:\n" + "\n".join(str(p) for p in missing))
    return pdf_to_md, md_to_tex, tex_to_json


def ensure_book_double_scripts_exist(script_dir: Path) -> tuple[Path, Path]:
    md_to_tex_book = script_dir / "src" / "book" / "mdTotex_book.py"
    merge_tex_blocks = script_dir / "src" / "book" / "merge_tex_blocks.py"
    missing = [p for p in [md_to_tex_book, merge_tex_blocks] if not p.exists()]
    if missing:
        raise SystemExit("Missing scripts:\n" + "\n".join(str(p) for p in missing))
    return md_to_tex_book, merge_tex_blocks


def ensure_naturalize_script_exists(script_dir: Path) -> Path:
    p = script_dir / "src" / "book" / "jsonNaturalize.py"
    if not p.exists():
        raise SystemExit(f"Missing script: {p}")
    return p


def _tmp_path(final_path: Path) -> Path:
    return final_path.with_name(final_path.name + ".tmp")


def _file_nonempty(p: Path) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size > 0


_PAGE_SENTINEL_RE = re.compile(r"(?m)^\s*<!--\s*PAGE\s+(\d+)\s*-->\s*$")


def _pdf_page_count(pdf_path: Path) -> Optional[int]:
    """Best-effort PDF page count using PyMuPDF (fitz)."""
    try:
        import fitz  # type: ignore
    except Exception:
        return None
    try:
        doc = fitz.open(str(pdf_path))
        try:
            return int(doc.page_count)
        finally:
            doc.close()
    except Exception:
        return None


def md_complete(md_path: Path, pdf_path: Path) -> bool:
    """MD is complete iff it contains the last PAGE sentinel matching PDF page count."""
    if not _file_nonempty(md_path):
        return False
    if not STRICT_RESUME:
        return True

    # Read a tail window first (fast for large files); fall back to full read if needed.
    try:
        tail_bytes = 1024 * 1024  # 1 MiB
        with md_path.open("rb") as f:
            try:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - tail_bytes), 0)
            except Exception:
                # Some file-like implementations may not support seek/tell well; fall back.
                f.seek(0)
            chunk = f.read()
        tail_text = chunk.decode("utf-8", errors="ignore")
        pages = [int(x) for x in _PAGE_SENTINEL_RE.findall(tail_text)]
        if not pages:
            # fallback: full file read
            text = md_path.read_text(encoding="utf-8", errors="ignore")
            pages = [int(x) for x in _PAGE_SENTINEL_RE.findall(text)]
    except Exception:
        return False
    if not pages:
        # If the OCR script didn't emit page sentinels, fall back to non-empty.
        return True

    max_md_page = max(pages)
    n_pdf = _pdf_page_count(pdf_path)
    if n_pdf is None:
        # can't confirm; accept as complete
        return True

    return max_md_page >= n_pdf


def tex_complete(tex_path: Path) -> bool:
    """TEX is complete iff it has \begin{document} and ends with \end{document} (best-effort)."""
    if not _file_nonempty(tex_path):
        return False
    if not STRICT_RESUME:
        return True

    try:
        head_bytes = 64 * 1024  # 64 KiB
        tail_bytes = 64 * 1024  # 64 KiB

        with tex_path.open("rb") as f:
            head = f.read(head_bytes)

            try:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - tail_bytes), 0)
            except Exception:
                # fall back: if seek/tell fails, just keep reading (small files)
                pass

            tail = f.read()

        head_text = head.decode("utf-8", errors="ignore")
        tail_text = tail.decode("utf-8", errors="ignore")
    except Exception:
        return False

    if "\\begin{document}" not in head_text and "\\begin{document}" not in tail_text:
        return False

    # Allow trailing whitespace/comments after \end{document}
    if re.search(r"\\end\{document\}\s*\Z", tail_text) is None:
        return False

    return True


def json_complete(json_path: Path) -> bool:
    """JSON is complete iff it parses and is a non-empty list/dict."""
    if not _file_nonempty(json_path):
        return False
    if not STRICT_RESUME:
        return True

    try:
        obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if isinstance(obj, list):
        return len(obj) > 0
    if isinstance(obj, dict):
        return len(obj) > 0
    return False


def naturalized_complete(json_path: Path) -> bool:
    return json_complete(json_path)


def _read_stats_json(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, int] = {}
    for k in ("rows", "llm_touched", "ok", "fallback", "skipped", "failed"):
        try:
            out[k] = int(obj.get(k, 0))
        except Exception:
            out[k] = 0
    return out


Validator = Callable[[Path], bool]


def run_stage_atomic(
    *,
    cmd: list[str],
    out_path: Path,
    validate_out: Validator,
) -> None:
    tmp = _tmp_path(out_path)
    if CLEAN_STALE_TMPS and tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    if len(cmd) < 4:
        raise RuntimeError("cmd too short (expected: python script IN OUT [flags...])")
    cmd2 = list(cmd)
    cmd2[3] = str(tmp)

    run_cmd(cmd2)

    if not validate_out(tmp):
        raise SystemExit(f"Stage produced incomplete output: {tmp}")

    tmp.replace(out_path)


def process_one(
    pdf_path: Path,
    pdf_to_md: Path,
    md_to_tex: Path,
    tex_to_json: Path,
    *,
    md_to_tex_book: Optional[Path] = None,
    merge_tex_blocks: Optional[Path] = None,
    mode: str,
    input_pdf_dir: Path,
    output_json_dir: Path,
    work_dir: Path,
) -> Tuple[Path, Optional[Dict[str, int]]]:
    rel_pdf = pdf_path.relative_to(input_pdf_dir)
    rel_no_suffix = rel_pdf.with_suffix("")

    job_dir = work_dir / rel_no_suffix
    job_dir.mkdir(parents=True, exist_ok=True)

    stem = pdf_path.stem
    md_path = job_dir / f"{stem}.md"
    tex_path = job_dir / f"{stem}.tex"

    json_path = (output_json_dir / rel_pdf).with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)

    naturalized_path = (OUTPUT_JSON_NATURALIZED_DIR / rel_pdf).with_suffix(".json")
    naturalized_path.parent.mkdir(parents=True, exist_ok=True)
    naturalize_stats_path = job_dir / f"{stem}.naturalize.stats.json"

    needs_naturalize = (
        ENABLE_NATURALIZE
        and mode == "book"
        and (
            NATURALIZE_FORCE
            or (not naturalized_complete(naturalized_path))
        )
    )

    if json_complete(json_path) and not OVERWRITE_JSON:
        print(f"[skip] {rel_pdf.as_posix()} -> JSON exists: {json_path}")
        if needs_naturalize:
            json_naturalize = ensure_naturalize_script_exists(PROJECT_ROOT)
            cmd4 = [sys.executable, str(json_naturalize), str(json_path), str(naturalized_path)]
            if NATURALIZE_MODEL:
                cmd4 += ["--model", str(NATURALIZE_MODEL)]
            cmd4 += ["--max-tokens", str(int(NATURALIZE_MAX_TOKENS))]
            cmd4 += ["--max-items", str(int(NATURALIZE_MAX_ITEMS))]
            if NATURALIZE_PROMPT_VERSION:
                cmd4 += ["--prompt-version", str(NATURALIZE_PROMPT_VERSION)]
            if NATURALIZE_CACHE_DIR:
                cmd4 += ["--cache-dir", str(NATURALIZE_CACHE_DIR)]
            if NATURALIZE_NO_CACHE:
                cmd4 += ["--no-cache"]
            if NATURALIZE_DISABLE_LLM:
                cmd4 += ["--disable-llm"]
            if NATURALIZE_FORCE:
                cmd4 += ["--force"]
            if not NATURALIZE_CLEAN_CACHE_ON_EXIT:
                cmd4 += ["--no-cleanup-cache-on-exit"]
            cmd4 += ["--stats-out", str(naturalize_stats_path)]

            run_cmd(cmd4)
            if not naturalized_complete(naturalized_path):
                raise SystemExit(f"Naturalize output incomplete: {naturalized_path}")
            return json_path, _read_stats_json(naturalize_stats_path)
        return json_path, None

    if md_complete(md_path, pdf_path):
        print(f"[resume] skip PDF->MD (complete): {md_path}")
    else:
        cmd1 = [sys.executable, str(pdf_to_md), str(pdf_path), str(md_path)]
        if OCR_MAX_TOKENS is not None:
            cmd1 += ["--max-tokens", str(OCR_MAX_TOKENS)]
        if OCR_WORKERS is not None:
            cmd1 += ["--workers", str(int(OCR_WORKERS))]

        if ATOMIC_OUTPUTS:
            run_stage_atomic(
                cmd=cmd1,
                out_path=md_path,
                validate_out=lambda p: md_complete(p, pdf_path),
            )
        else:
            run_cmd(cmd1)
            if not md_complete(md_path, pdf_path):
                raise SystemExit(f"PDF->MD output incomplete: {md_path}")

    if tex_complete(tex_path):
        print(f"[resume] skip MD->TEX (complete): {tex_path}")
    else:
        cmd2 = [sys.executable, str(md_to_tex), str(md_path), str(tex_path)]
        if THINK_WORKERS is not None:
            cmd2 += ["--workers", str(int(THINK_WORKERS))]
        if MDTOTEX_SKIP_TAG_RECOVERY:
            # Force mdTotex to skip page-wise tag recovery.
            cmd2 += ["--pdf", "/dev/null"]

        if ATOMIC_OUTPUTS:
            run_stage_atomic(
                cmd=cmd2,
                out_path=tex_path,
                validate_out=tex_complete,
            )
        else:
            run_cmd(cmd2)
            if not tex_complete(tex_path):
                raise SystemExit(f"MD->TEX output incomplete: {tex_path}")

    if mode == "book" and BOOK_ENABLE_DOUBLE_ROUTE:
        if md_to_tex_book is None or merge_tex_blocks is None:
            raise SystemExit("book double-route enabled but mdTotex_book/merge_tex_blocks script is missing")

        rel_mode_pdf = pdf_path.relative_to(input_pdf_dir / mode)
        double_root = work_dir / mode / BOOK_DOUBLE_DIRNAME
        double_tex_path = (double_root / rel_mode_pdf).with_suffix(".tex")
        double_tex_path.parent.mkdir(parents=True, exist_ok=True)

        if tex_complete(double_tex_path):
            print(f"[resume] skip MD->TEX(book-double) (complete): {double_tex_path}")
        else:
            cmd2b = [sys.executable, str(md_to_tex_book), str(md_path), str(double_tex_path)]
            if THINK_WORKERS is not None:
                cmd2b += ["--workers", str(int(THINK_WORKERS))]
            if MDTOTEX_SKIP_TAG_RECOVERY:
                cmd2b += ["--pdf", "/dev/null"]

            if ATOMIC_OUTPUTS:
                run_stage_atomic(
                    cmd=cmd2b,
                    out_path=double_tex_path,
                    validate_out=tex_complete,
                )
            else:
                run_cmd(cmd2b)
                if not tex_complete(double_tex_path):
                    raise SystemExit(f"MD->TEX(book-double) output incomplete: {double_tex_path}")

        cmd_merge = [
            sys.executable,
            str(merge_tex_blocks),
            "--from-tex",
            str(double_tex_path),
            "--to-tex",
            str(tex_path),
        ]
        if BOOK_MERGE_NO_DEDUPE:
            cmd_merge += ["--no-dedupe"]
        if BOOK_MERGE_INCLUDE_TYPES:
            cmd_merge += ["--include-types", str(BOOK_MERGE_INCLUDE_TYPES)]
        if BOOK_MERGE_EXCLUDE_TYPES:
            cmd_merge += ["--exclude-types", str(BOOK_MERGE_EXCLUDE_TYPES)]
        if BOOK_MERGE_POSITION:
            cmd_merge += ["--position", str(BOOK_MERGE_POSITION)]

        run_cmd(cmd_merge)
        if not tex_complete(tex_path):
            raise SystemExit(f"Merged TEX output incomplete: {tex_path}")

    if json_complete(json_path) and not OVERWRITE_JSON:
        print(f"[resume] skip TEX->JSON (complete): {json_path}")
        return json_path, None

    cmd3 = [sys.executable, str(tex_to_json), str(tex_path), str(json_path)]
    # Carry per-PDF source name into JSON rows (e.g. "book/foo/bar").
    cmd3 += ["--source-name", rel_no_suffix.as_posix()]
    if TEX_WARNINGS_JSON:
        cmd3 += ["--warnings-json", str(job_dir / f"{stem}.warnings.json")]
    if TEX_EMIT_RAW:
        cmd3 += ["--emit-raw"]
    if TEX_TYPE_REFINE:
        cmd3 += ["--type-refine"]
    if TEX_NO_DIRECT_ANSWER:
        cmd3 += ["--no-direct-answer"]
    if TEX_LLM_SELF_CHECK:
        cmd3 += ["--llm-self-check"]
    if TEX_LLM_MODEL:
        cmd3 += ["--llm-model", str(TEX_LLM_MODEL)]
    cmd3 += ["--llm-max-items", str(int(TEX_LLM_MAX_ITEMS))]
    cmd3 += ["--llm-max-rounds", str(int(TEX_LLM_MAX_ROUNDS))]
    if TEX_LLM_CACHE_DIR:
        cmd3 += ["--llm-cache-dir", str(TEX_LLM_CACHE_DIR)]
    if TEX_LLM_NO_CACHE:
        cmd3 += ["--llm-no-cache"]
    if TEX_LLM_TYPE_CHECK:
        cmd3 += ["--llm-type-check"]
    cmd3 += ["--llm-type-max-items", str(int(TEX_LLM_TYPE_MAX_ITEMS))]
    if TEX_LLM_DIRECT_ANSWER_FALLBACK:
        cmd3 += ["--llm-direct-answer-fallback"]
    cmd3 += ["--llm-direct-answer-max-items", str(int(TEX_LLM_DIRECT_ANSWER_MAX_ITEMS))]
    if TEX_REQUIRE_CLEAN:
        cmd3 += ["--require-clean"]
    if TEX_FINAL_JSON_ONLY:
        cmd3 += ["--final-json-only"]

    if ATOMIC_OUTPUTS:
        run_stage_atomic(
            cmd=cmd3,
            out_path=json_path,
            validate_out=json_complete,
        )
    else:
        run_cmd(cmd3)
        if not json_complete(json_path):
            raise SystemExit(f"TEX->JSON output incomplete: {json_path}")

    if ENABLE_NATURALIZE and mode == "book":
        json_naturalize = ensure_naturalize_script_exists(PROJECT_ROOT)
        cmd4 = [sys.executable, str(json_naturalize), str(json_path), str(naturalized_path)]
        if NATURALIZE_MODEL:
            cmd4 += ["--model", str(NATURALIZE_MODEL)]
        cmd4 += ["--max-tokens", str(int(NATURALIZE_MAX_TOKENS))]
        cmd4 += ["--max-items", str(int(NATURALIZE_MAX_ITEMS))]
        if NATURALIZE_PROMPT_VERSION:
            cmd4 += ["--prompt-version", str(NATURALIZE_PROMPT_VERSION)]
        if NATURALIZE_CACHE_DIR:
            cmd4 += ["--cache-dir", str(NATURALIZE_CACHE_DIR)]
        if NATURALIZE_NO_CACHE:
            cmd4 += ["--no-cache"]
        if NATURALIZE_DISABLE_LLM:
            cmd4 += ["--disable-llm"]
        if NATURALIZE_FORCE:
            cmd4 += ["--force"]
        if not NATURALIZE_CLEAN_CACHE_ON_EXIT:
            cmd4 += ["--no-cleanup-cache-on-exit"]
        cmd4 += ["--stats-out", str(naturalize_stats_path)]

        run_cmd(cmd4)
        if not naturalized_complete(naturalized_path):
            raise SystemExit(f"Naturalize output incomplete: {naturalized_path}")
        return json_path, _read_stats_json(naturalize_stats_path)

    return json_path, None


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    ap = argparse.ArgumentParser(description="Run the OCR -> Markdown -> TeX -> JSON pipeline.")
    ap.add_argument(
        "--mode",
        choices=["book", "paper"],
        default=str(get_setting(settings, "PIPELINE_MODE", "book")),
        help="Pipeline mode to run (default: %(default)s)",
    )
    return ap.parse_args()


def main() -> None:
    global INPUT_PDF_DIR, OUTPUT_JSON_DIR, WORK_DIR
    global PIPELINE_MODE, OCR_MAX_TOKENS, ONLY_THESE_STEMS, OVERWRITE_JSON
    global OCR_WORKERS, THINK_WORKERS, STRICT_RESUME, ATOMIC_OUTPUTS, CLEAN_STALE_TMPS
    global TEX_WARNINGS_JSON, TEX_EMIT_RAW, TEX_TYPE_REFINE, TEX_NO_DIRECT_ANSWER
    global TEX_LLM_SELF_CHECK, TEX_LLM_MODEL, TEX_LLM_MAX_ITEMS, TEX_LLM_MAX_ROUNDS
    global TEX_LLM_CACHE_DIR, TEX_LLM_NO_CACHE
    global TEX_LLM_TYPE_CHECK, TEX_LLM_TYPE_MAX_ITEMS
    global TEX_LLM_DIRECT_ANSWER_FALLBACK, TEX_LLM_DIRECT_ANSWER_MAX_ITEMS
    global TEX_REQUIRE_CLEAN, TEX_FINAL_JSON_ONLY
    global MDTOTEX_SKIP_TAG_RECOVERY
    global BOOK_ENABLE_DOUBLE_ROUTE, BOOK_DOUBLE_DIRNAME
    global BOOK_MERGE_NO_DEDUPE, BOOK_MERGE_INCLUDE_TYPES, BOOK_MERGE_EXCLUDE_TYPES, BOOK_MERGE_POSITION
    global OUTPUT_JSON_NATURALIZED_DIR, ENABLE_NATURALIZE
    global NATURALIZE_MODEL, NATURALIZE_MAX_TOKENS, NATURALIZE_MAX_ITEMS
    global NATURALIZE_PROMPT_VERSION, NATURALIZE_CACHE_DIR
    global NATURALIZE_NO_CACHE, NATURALIZE_DISABLE_LLM, NATURALIZE_FORCE, NATURALIZE_CLEAN_CACHE_ON_EXIT

    settings = load_settings()
    INPUT_PDF_DIR = PROJECT_ROOT / str(get_setting(settings, "INPUT_PDF_DIR", "input_pdfs"))
    OUTPUT_JSON_DIR = PROJECT_ROOT / str(get_setting(settings, "OUTPUT_JSON_DIR", "output_json"))
    OUTPUT_JSON_NATURALIZED_DIR = PROJECT_ROOT / str(
        get_setting(settings, "OUTPUT_JSON_NATURALIZED_DIR", "output_json_naturalized")
    )
    WORK_DIR = PROJECT_ROOT / str(get_setting(settings, "WORK_DIR", "work"))
    PIPELINE_MODE = str(get_setting(settings, "PIPELINE_MODE", "book"))
    OCR_MAX_TOKENS = get_setting(settings, "OCR_MAX_TOKENS", None)
    ONLY_THESE_STEMS = set(get_setting(settings, "ONLY_THESE_STEMS", []))
    OVERWRITE_JSON = bool(get_setting(settings, "OVERWRITE_JSON", False))
    OCR_WORKERS = get_setting(settings, "OCR_WORKERS", 4)
    THINK_WORKERS = get_setting(settings, "THINK_WORKERS", 4)
    STRICT_RESUME = bool(get_setting(settings, "STRICT_RESUME", True))
    ATOMIC_OUTPUTS = bool(get_setting(settings, "ATOMIC_OUTPUTS", True))
    CLEAN_STALE_TMPS = bool(get_setting(settings, "CLEAN_STALE_TMPS", True))
    TEX_WARNINGS_JSON = bool(get_setting(settings, "TEX_WARNINGS_JSON", False))
    TEX_EMIT_RAW = bool(get_setting(settings, "TEX_EMIT_RAW", False))
    TEX_TYPE_REFINE = bool(get_setting(settings, "TEX_TYPE_REFINE", False))
    TEX_NO_DIRECT_ANSWER = bool(get_setting(settings, "TEX_NO_DIRECT_ANSWER", False))
    TEX_LLM_SELF_CHECK = bool(get_setting(settings, "TEX_LLM_SELF_CHECK", False))
    TEX_LLM_MODEL = str(get_setting(settings, "TEX_LLM_MODEL", "gpt-5-mini"))
    TEX_LLM_MAX_ITEMS = int(get_setting(settings, "TEX_LLM_MAX_ITEMS", 20))
    TEX_LLM_MAX_ROUNDS = int(get_setting(settings, "TEX_LLM_MAX_ROUNDS", 3))
    TEX_LLM_CACHE_DIR = str(get_setting(settings, "TEX_LLM_CACHE_DIR", ""))
    TEX_LLM_NO_CACHE = bool(get_setting(settings, "TEX_LLM_NO_CACHE", False))
    TEX_LLM_TYPE_CHECK = bool(get_setting(settings, "TEX_LLM_TYPE_CHECK", False))
    TEX_LLM_TYPE_MAX_ITEMS = int(get_setting(settings, "TEX_LLM_TYPE_MAX_ITEMS", 200))
    TEX_LLM_DIRECT_ANSWER_FALLBACK = bool(get_setting(settings, "TEX_LLM_DIRECT_ANSWER_FALLBACK", False))
    TEX_LLM_DIRECT_ANSWER_MAX_ITEMS = int(get_setting(settings, "TEX_LLM_DIRECT_ANSWER_MAX_ITEMS", 0))
    TEX_REQUIRE_CLEAN = bool(get_setting(settings, "TEX_REQUIRE_CLEAN", False))
    TEX_FINAL_JSON_ONLY = bool(get_setting(settings, "TEX_FINAL_JSON_ONLY", False))
    MDTOTEX_SKIP_TAG_RECOVERY = bool(get_setting(settings, "MDTOTEX_SKIP_TAG_RECOVERY", False))
    BOOK_ENABLE_DOUBLE_ROUTE = bool(get_setting(settings, "BOOK_ENABLE_DOUBLE_ROUTE", True))
    BOOK_DOUBLE_DIRNAME = str(get_setting(settings, "BOOK_DOUBLE_DIRNAME", "double_book"))
    BOOK_MERGE_NO_DEDUPE = bool(get_setting(settings, "BOOK_MERGE_NO_DEDUPE", False))
    BOOK_MERGE_INCLUDE_TYPES = str(get_setting(settings, "BOOK_MERGE_INCLUDE_TYPES", ""))
    BOOK_MERGE_EXCLUDE_TYPES = str(get_setting(settings, "BOOK_MERGE_EXCLUDE_TYPES", ""))
    BOOK_MERGE_POSITION = str(get_setting(settings, "BOOK_MERGE_POSITION", "before-end-document"))
    ENABLE_NATURALIZE = bool(get_setting(settings, "ENABLE_NATURALIZE", False))
    NATURALIZE_MODEL = str(get_setting(settings, "NATURALIZE_MODEL", ""))
    NATURALIZE_MAX_TOKENS = int(get_setting(settings, "NATURALIZE_MAX_TOKENS", 900))
    NATURALIZE_MAX_ITEMS = int(get_setting(settings, "NATURALIZE_MAX_ITEMS", 0))
    NATURALIZE_PROMPT_VERSION = str(get_setting(settings, "NATURALIZE_PROMPT_VERSION", "v1"))
    NATURALIZE_CACHE_DIR = str(get_setting(settings, "NATURALIZE_CACHE_DIR", ""))
    NATURALIZE_NO_CACHE = bool(get_setting(settings, "NATURALIZE_NO_CACHE", False))
    NATURALIZE_DISABLE_LLM = bool(get_setting(settings, "NATURALIZE_DISABLE_LLM", False))
    NATURALIZE_FORCE = bool(get_setting(settings, "NATURALIZE_FORCE", False))
    NATURALIZE_CLEAN_CACHE_ON_EXIT = bool(get_setting(settings, "NATURALIZE_CLEAN_CACHE_ON_EXIT", True))

    args = parse_args()
    mode = args.mode

    INPUT_PDF_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_NATURALIZED_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    pdf_to_md, md_to_tex, tex_to_json = ensure_scripts_exist(PROJECT_ROOT, mode)
    md_to_tex_book: Optional[Path] = None
    merge_tex_blocks: Optional[Path] = None
    if mode == "book" and BOOK_ENABLE_DOUBLE_ROUTE:
        md_to_tex_book, merge_tex_blocks = ensure_book_double_scripts_exist(PROJECT_ROOT)

    mode_input_dir = INPUT_PDF_DIR / mode
    mode_input_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(mode_input_dir.rglob("*.pdf"))

    if ONLY_THESE_STEMS:
        raw_selectors = list(ONLY_THESE_STEMS)

        norm: List[str] = []
        for s in raw_selectors:
            t = s.strip().replace("\\", "/")
            if t.endswith(".pdf"):
                t = t[:-4]
            norm.append(t)

        def _matches(p: Path) -> bool:
            rel = p.relative_to(INPUT_PDF_DIR)
            rel_no_suffix_posix = rel.with_suffix("").as_posix()

            for sel in norm:
                if not sel:
                    continue
                sel_strip = sel.rstrip("/")

                if p.stem == sel_strip:
                    return True

                if rel_no_suffix_posix == sel_strip:
                    return True

                if rel_no_suffix_posix.startswith(sel_strip + "/"):
                    return True

            return False

        pdfs = [p for p in pdfs if _matches(p)]

    def _json_out_path(p: Path) -> Path:
        rel = p.relative_to(INPUT_PDF_DIR)
        return (OUTPUT_JSON_DIR / rel).with_suffix(".json")

    def _json_nat_out_path(p: Path) -> Path:
        rel = p.relative_to(INPUT_PDF_DIR)
        return (OUTPUT_JSON_NATURALIZED_DIR / rel).with_suffix(".json")

    if not OVERWRITE_JSON:
        def _needs_work(p: Path) -> bool:
            j1 = _json_out_path(p)
            if not json_complete(j1):
                return True
            if ENABLE_NATURALIZE and mode == "book":
                j2 = _json_nat_out_path(p)
                if NATURALIZE_FORCE or (not naturalized_complete(j2)):
                    return True
            return False

        pdfs = [p for p in pdfs if _needs_work(p)]

    if not pdfs:
        print(f"No PDFs to process for mode={mode!r} in: {mode_input_dir}")
        return

    print(f"Mode: {mode}")
    print(f"Found {len(pdfs)} PDF(s) to process in {mode_input_dir}")
    naturalize_totals: Dict[str, int] = {
        "rows": 0,
        "llm_touched": 0,
        "ok": 0,
        "fallback": 0,
        "skipped": 0,
        "failed": 0,
    }
    for pdf in pdfs:
        print(f"\n=== Processing: {pdf.relative_to(INPUT_PDF_DIR).as_posix()} ===")
        out_json, nat_stats = process_one(
            pdf,
            pdf_to_md,
            md_to_tex,
            tex_to_json,
            md_to_tex_book=md_to_tex_book,
            merge_tex_blocks=merge_tex_blocks,
            mode=mode,
            input_pdf_dir=INPUT_PDF_DIR,
            output_json_dir=OUTPUT_JSON_DIR,
            work_dir=WORK_DIR,
        )
        print(f"[ok] JSON -> {out_json}")
        if nat_stats:
            for k in naturalize_totals:
                naturalize_totals[k] += int(nat_stats.get(k, 0))

    if ENABLE_NATURALIZE and mode == "book":
        print(
            "[naturalize-summary] "
            f"rows={naturalize_totals['rows']}, "
            f"llm_touched={naturalize_totals['llm_touched']}, "
            f"ok={naturalize_totals['ok']}, "
            f"fallback={naturalize_totals['fallback']}, "
            f"skipped={naturalize_totals['skipped']}, "
            f"failed={naturalize_totals['failed']}"
        )
    print("\nALL DONE")


if __name__ == "__main__":
    main()
