#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Step-2 naturalization: rewrite step-1 problem JSON into standardized math wording.

Input:  step-1 JSON rows (from texTojson.py)
Output: same rows + step-2 fields:
  - problem_standardized_math
  - problem_finally
  - naturalize_status
  - naturalize_prompt_version
  - naturalize_notes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm


_CHAT_FORCE_STREAM: Optional[bool] = None
ALLOWED_STATUS = {"ok", "fallback_original", "fallback_context", "skipped", "failed"}
PROMPT_VERSION_DEFAULT = "v1"
PROBLEM_FINALLY_PROMPT_VERSION_DEFAULT = "v1"


def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    try:
        v = json.loads(s)
    except Exception:
        return None
    return v if isinstance(v, dict) else None


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
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


def _is_stream_required_error(err: Exception) -> bool:
    return "stream must be set to true" in str(err or "").lower()


def _is_chat_not_supported_error(err: Exception) -> bool:
    msg = str(err or "").lower()
    return (
        "chat/completions" in msg
        or ("405" in msg and "not allowed" in msg)
        or "method not allowed" in msg
    )


def _is_max_tokens_unsupported_error(err: Exception) -> bool:
    msg = str(err or "").lower()
    return (
        "unsupported parameter" in msg
        and "max_tokens" in msg
        and "max_completion_tokens" in msg
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


def _collect_responses_text(resp: Any) -> str:
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    parts: List[str] = []
    for item in (getattr(resp, "output", None) or []):
        contents = getattr(item, "content", None)
        if contents is None and isinstance(item, dict):
            contents = item.get("content")
        for c in (contents or []):
            if isinstance(c, dict):
                t = c.get("text") or ""
            else:
                t = getattr(c, "text", "") or ""
            if isinstance(t, str) and t:
                parts.append(t)
    return "".join(parts).strip()


def _responses_completion_text(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
) -> str:
    kwargs = dict(
        model=model,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
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
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    def _with_completion_tokens(base: dict) -> dict:
        k = dict(base)
        if "max_tokens" in k:
            k.pop("max_tokens", None)
            k["max_completion_tokens"] = max_tokens
        return k

    if _CHAT_FORCE_STREAM is True:
        try:
            stream_obj = client.chat.completions.create(stream=True, **kwargs)
            return (_collect_stream_text(stream_obj) or "").strip()
        except Exception as e:
            if _is_max_tokens_unsupported_error(e):
                stream_obj = client.chat.completions.create(stream=True, **_with_completion_tokens(kwargs))
                return (_collect_stream_text(stream_obj) or "").strip()
            if _is_chat_not_supported_error(e):
                return (_responses_completion_text(client, model=model, prompt=prompt, max_tokens=max_tokens) or "").strip()
            raise
    try:
        resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        if _is_max_tokens_unsupported_error(e):
            try:
                resp = client.chat.completions.create(**_with_completion_tokens(kwargs))
                return (resp.choices[0].message.content or "").strip()
            except Exception as e_mt:
                e = e_mt
        if _is_stream_required_error(e):
            _CHAT_FORCE_STREAM = True
            try:
                stream_obj = client.chat.completions.create(stream=True, **kwargs)
                return (_collect_stream_text(stream_obj) or "").strip()
            except Exception as e2:
                if _is_max_tokens_unsupported_error(e2):
                    stream_obj = client.chat.completions.create(stream=True, **_with_completion_tokens(kwargs))
                    return (_collect_stream_text(stream_obj) or "").strip()
                if _is_chat_not_supported_error(e2):
                    return (_responses_completion_text(client, model=model, prompt=prompt, max_tokens=max_tokens) or "").strip()
                raise
        if _is_chat_not_supported_error(e):
            return (_responses_completion_text(client, model=model, prompt=prompt, max_tokens=max_tokens) or "").strip()

        # Generic fallback for gateways that fail non-stream chat with empty/non-JSON bodies.
        _CHAT_FORCE_STREAM = True
        try:
            stream_obj = client.chat.completions.create(stream=True, **kwargs)
            return (_collect_stream_text(stream_obj) or "").strip()
        except Exception as e3:
            if _is_max_tokens_unsupported_error(e3):
                stream_obj = client.chat.completions.create(stream=True, **_with_completion_tokens(kwargs))
                return (_collect_stream_text(stream_obj) or "").strip()
            if _is_chat_not_supported_error(e3):
                return (_responses_completion_text(client, model=model, prompt=prompt, max_tokens=max_tokens) or "").strip()
            raise e


def _default_cache_dir() -> Path:
    d = Path.cwd() / "cache" / "naturalize"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _call_cached(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    cache_dir: Optional[Path],
    cache_enabled: bool,
) -> str:
    if not cache_enabled:
        return _chat_completion_text(client, model=model, prompt=prompt, max_tokens=max_tokens)

    cdir = cache_dir or _default_cache_dir()
    key = hashlib.sha256((model + "\n" + str(max_tokens) + "\n" + prompt).encode("utf-8")).hexdigest()
    path = cdir / f"{key}.txt"
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            pass
    out = _chat_completion_text(client, model=model, prompt=prompt, max_tokens=max_tokens)
    try:
        path.write_text(out, encoding="utf-8")
    except Exception:
        pass
    return out


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


def _compact_text(s: str, max_chars: int = 320) -> str:
    t = re.sub(r"\s+", " ", s or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "…"


def _extract_context_reference_lines(problem_with_context: str) -> List[str]:
    """
    Extract all reference lines before the first "Problem:" line.
    """
    text = str(problem_with_context or "")
    if not text.strip():
        return []
    lines = [ln.strip() for ln in text.splitlines()]
    out: List[str] = []
    for ln in lines:
        if re.match(r"^\s*Problem\s*:", ln, flags=re.IGNORECASE):
            break
        if ln:
            # Clean display-only line-break markers inserted upstream.
            ln = re.sub(r"\s*\\\s*$", "", ln).strip()
        if ln:
            out.append(ln)
    return out



def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for it in items:
        k = (it or "").strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out

def build_naturalize_input(record: Dict[str, Any]) -> Dict[str, Any]:
    dep = record.get("dependency")
    dep_obj = dep if isinstance(dep, dict) else {}

    source_idx = str(record.get("source_idx") or dep_obj.get("source_idx") or "").strip()
    source = str(record.get("source") or dep_obj.get("source") or "").strip()
    original_problem = str(record.get("problem") or "").strip()
    problem_with_context = _normalize_tex_math_text(str(record.get("problem_with_context") or dep_obj.get("problem_with_context") or "").strip())

    # Only keep direct dependency tags recognized from the problem itself.
    # Do not inherit expanded/composite tags from referenced theorem/algorithm bodies.
    targets = record.get("body_ref_targets")
    if not isinstance(targets, list):
        targets = dep_obj.get("body_ref_targets")
    if not isinstance(targets, list):
        targets = []

    equations: List[Dict[str, str]] = []
    contexts: List[Dict[str, str]] = []
    seen_tag_keys: set[str] = set()
    for t in targets:
        if not isinstance(t, dict):
            continue
        tag = str(t.get("tag") or "").strip()
        display_tag = str(t.get("display_tag") or "").strip()
        eq = str(t.get("equation_content") or "").strip()
        ctx = str(t.get("content") or "").strip()

        # Inherit each recognized tag at most once.
        key = tag or display_tag
        if key:
            if key in seen_tag_keys:
                continue
            seen_tag_keys.add(key)

        if eq:
            equations.append(
                {
                    "tag": tag,
                    "display_tag": display_tag,
                    "equation_content": eq,
                }
            )
        if ctx:
            contexts.append(
                {
                    "tag": tag,
                    "display_tag": display_tag,
                    "content": _compact_text(ctx, max_chars=320),
                }
            )
    context_references = _dedupe_keep_order(_extract_context_reference_lines(problem_with_context))

    return {
        "source_idx": source_idx,
        "source": source,
        "original_problem": original_problem,
        "problem_with_context": problem_with_context,
        "context_references": context_references,
        "equations": equations,
        "contexts": contexts,
    }


def make_prompt(payload: Dict[str, Any]) -> str:
    return (
        "You are given ONE mathematical exercise record extracted from a textbook.\n"
        "Your job is to rewrite it into standard mathematical English while staying strictly faithful to the original statement.\n\n"
        "Priority of information:\n"
        "1. original_problem is the primary source.\n"
        "2. context_references/problem_with_context provide reference formulas/theorems/algorithms that MUST be integrated when clearly relevant.\n"
        "3. equations contains formulas that must be preserved when they belong to the exercise.\n"
        "4. contexts can help resolve nearby references, but you must not invent content from it.\n\n"
        "Integration rule for references:\n"
        "- MUST incorporate all clearly relevant referenced conditions/formulas from context_references/problem_with_context into the standardized problem statement.\n"
        "- MUST resolve local pointers like 'Theorem 11.8', '(11.46)', 'Algorithm 11.5' into explicit mathematical wording when support is present in input.\n"
        "- If reference content is available, DO NOT write pointer-only phrases such as 'assume the hypotheses of Theorem X.Y'; instead, spell out those hypotheses explicitly.\n"
        "- If a referenced formula/algorithm text is available, inline the concrete mathematical statement or procedure condition that is needed for the exercise.\n"
        "- Avoid leaving bare citation markers as substitutes for mathematical content.\n"
        "- If a reference is ambiguous or unsupported, keep pointer text and do NOT invent details.\n\n"
        "Hard constraints:\n"
        "- Do NOT solve the problem.\n"
        "- Do NOT add any assumptions, definitions, domains, properties, or new symbols that are not explicit in the input.\n"
        "- Do NOT add new tasks/questions/sub-questions.\n"
        "- Do NOT change task type (prove/show/find/derive/evaluate/minimize/etc.).\n"
        "- Do NOT remove essential formulas, quantifiers, constraints, variables, domains, or conditions.\n"
        "- Keep the original meaning and task unchanged; only standardize wording and notation.\n"
        "- Remove non-mathematical filler and redundant narrative words, but do NOT remove mathematical content.\n"
        "- Prefer concise proposition-style mathematical phrasing.\n"
        "- Use formal textbook-style English.\n"
        "- Prefer explicit mathematical statements over metareferences to numbered items.\n"
        "- Use STANDARD LaTeX math notation only; no Unicode math symbols.\n"
        "- Keep formulas compilable and avoid malformed delimiters.\n"
        "- Return STRICT JSON only, exactly one key.\n\n"
        "Additional requirements (append-only):\n"
        "- When references are available in input, explicitly inline the needed assumptions/formulas; avoid unresolved metareferences.\n"
        "- Keep semantic equivalence exactly: do not strengthen/weaken assumptions, claims, or quantifier scope.\n"
        "- Ensure all math uses standard TeX commands and valid delimiters.\n\n"
        "Output JSON schema:\n"
        '{"problem_standardized_math":"<faithful standardized mathematical exercise statement>"}\n\n'
        "Input JSON:\n"
        + json.dumps(payload, ensure_ascii=False)
    )

def make_retry_prompt(payload: Dict[str, Any]) -> str:
    return (
        "Your previous output was invalid. Try again.\n"
        "Return STRICT JSON ONLY for one mathematical exercise.\n\n"
        "Hard constraints:\n"
        "- English only.\n"
        "- Preserve original meaning, task type, notation, formulas, constraints, and quantifiers.\n"
        "- MUST integrate clearly relevant references from context_references/problem_with_context into the standardized statement when available.\n"
        "- Do NOT keep pointer-only wording (for example, 'assume the hypotheses of Theorem X.Y') when referenced content is available; expand it explicitly.\n"
        "- MUST replace reference-only mentions by concrete mathematical assumptions/formulas extracted from the input whenever possible.\n"
        "- Remove non-mathematical filler and redundant narrative words only.\n"
        "- Keep the output concise and proposition-style.\n"
        "- Do NOT add assumptions, definitions, domains, or new symbols.\n"
        "- Do NOT add new tasks/questions/sub-questions.\n"
        "- Do NOT solve the problem.\n"
        "- Do NOT output explanation, markdown, or extra keys.\n"
        "- Use STANDARD LaTeX math notation only; no Unicode math symbols.\n"
        "- Exactly one key: problem_standardized_math.\n\n"
        "Additional requirements (append-only):\n"
        "- Expand pronouns/metareferences (this theorem/result/algorithm/equation) into explicit math content when supported by input.\n"
        "- Keep final wording proposition-style and faithful to original task intent.\n"
        "- Keep TeX notation strict and compilable.\n\n"
        "Output template:\n"
        '{"problem_standardized_math":"<faithful standardized mathematical exercise statement>"}\n\n'
        "Input JSON:\n"
        + json.dumps(payload, ensure_ascii=False)
    )

def make_problem_complete_prompt(problem_standardized_math: str) -> str:
    payload = {"problem_standardized_math": problem_standardized_math}
    return (
        "You are given one optimization exercise statement.\n"
        "Rewrite it into a condition-complete, self-contained, textbook-style problem.\n\n"
        "Hard constraints:\n"
        "- Do NOT solve the problem.\n"
        "- Keep the original mathematical intent and task type.\n"
        "- Do NOT add unnecessary stronger assumptions.\n"
        "- Keep notation faithful and LaTeX-friendly.\n"
        "- Use STANDARD LaTeX math notation only (no Unicode math symbols).\n"
        "- If the exercise has multiple sub-questions, separate them as numbered items: 1. 2. 3.\n"
        "- For sets/spaces use \mathbf{R}, \mathbf{N}, \mathbf{Z}, \mathbf{Q}, \mathbf{C} when relevant.\n"
        "- Use LaTeX operators: \le, \ge, \ne, \in, \subseteq, \to, \times, \cdot, \nabla.\n"
        "- Keep all formulas compilable and avoid malformed TeX delimiters.\n"
        "- Use explicit LaTeX commands (for example \\alpha, \\le, \\times, \\mathbf{R}) instead of Unicode math symbols.\n"
        "- Keep math delimiters consistent: use \\( ... \\) / \\[ ... \\] correctly.\n"
        "- Return JSON only.\n\n"
        "Additional requirements (append-only):\n"
        "- Preserve the exact mathematical intent from problem_standardized_math; only improve completeness/readability.\n"
        "- Keep integrated dependency conditions explicit in mathematical form, not citation placeholders.\n"
        "- Do not introduce new symbols unless directly implied by existing notation.\n\n"
        "Output schema:\n"
        '{"problem_complete":"<condition-complete statement>"}\n\n'
        "Input:\n"
        + json.dumps(payload, ensure_ascii=False)
    )

def make_problem_concise_prompt(problem_complete: str) -> str:
    payload = {"problem_complete": problem_complete}
    return (
        "You are given one complete optimization problem statement.\n"
        "Rewrite it to be concise and formalization-friendly while preserving meaning.\n\n"
        "Hard constraints:\n"
        "- Do NOT solve the problem.\n"
        "- Keep all essential assumptions and constraints.\n"
        "- Remove narrative filler and keep formal mathematical phrasing.\n"
        "- If there are multiple tasks/sub-questions, split them into numbered items: 1. 2. 3.\n"
        "- Use STANDARD LaTeX math notation only (no Unicode math symbols).\n"
        "- Keep LaTeX commands explicit (for example \\alpha, \\mathbf{R}, \\times, \\le).\n"
        "- Keep notation consistent and compilable; preserve mathematical correctness.\n"
        "- Keep delimiters and escapes valid LaTeX (no malformed \\(, \\), \\[, \\]).\n"
        "- Prefer explicit operators \le, \ge, \ne, \in, \subseteq, \to, \times, \cdot.\n"
        "- Return JSON only.\n\n"
        "Additional requirements (append-only):\n"
        "- Keep semantic equivalence with problem_complete exactly.\n"
        "- Do not compress away dependency assumptions/formulas required by the exercise.\n"
        "- Keep TeX output standard and parser-stable.\n\n"
        "Output schema:\n"
        '{"problem_concise":"<concise statement>"}\n\n'
        "Input:\n"
        + json.dumps(payload, ensure_ascii=False)
    )

def _jsonish_unescape_preserve_latex(val: str) -> str:
    t = str(val or "")
    # Keep LaTeX commands intact: do NOT blindly map "\n" -> newline.
    t = t.replace('\\"', '"')
    t = re.sub(r"\\n(?![A-Za-z])", "\n", t)
    t = re.sub(r"\\t(?![A-Za-z])", "\t", t)
    t = re.sub(r"\\r(?![A-Za-z])", "\r", t)
    return t


def _extract_json_string_by_keys(raw: str, keys: List[str]) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    obj = _extract_first_json_object(text)
    if isinstance(obj, dict):
        for k in keys:
            val = str(obj.get(k) or "").strip()
            if val:
                return val

    for k in keys:
        m = re.search(rf'"{re.escape(k)}"\s*:\s*"([\s\S]*?)"\s*(?:[,}}])', text)
        if m:
            val = m.group(1)
            val = _jsonish_unescape_preserve_latex(val).strip()
            if val:
                return val

    text = re.sub(r"^\s*```(?:json|text)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```\s*$", "", text).strip()
    for k in keys:
        text = re.sub(rf"^\s*{re.escape(k)}\s*[:：]\s*", "", text, flags=re.IGNORECASE).strip()
    if text.startswith("{") and text.endswith("}"):
        return ""
    return text


def _extract_standardized_text(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    obj = _extract_first_json_object(text)
    if obj:
        val = str(obj.get("problem_standardized_math") or "").strip()
        if val:
            return val
    # Try key-value extraction from quasi-JSON output
    m = re.search(r'"problem_standardized_math"\s*:\s*"([\s\S]*?)"\s*(?:[,}])', text)
    if m:
        val = m.group(1)
        val = _jsonish_unescape_preserve_latex(val).strip()
        if val:
            return val
    # Fallback: strip code fences / leading labels and keep plain text
    text = re.sub(r"^\s*```(?:json|text)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```\s*$", "", text).strip()
    text = re.sub(r"^\s*problem_standardized_math\s*[:：]\s*", "", text, flags=re.IGNORECASE).strip()
    if text.startswith("{") and text.endswith("}"):
        return ""
    return text


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_EN_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_EN_ALPHA_RE = re.compile(r"[A-Za-z]")
_UNICODE_ALPHA_RE = re.compile(r"[^\W\d_]", flags=re.UNICODE)


def _looks_non_english(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return True
    return bool(_CJK_RE.search(t))


def _english_quality_ok(s: str, *, min_words: int = 6, min_alpha_ratio: float = 0.6) -> bool:
    t = (s or "").strip()
    if not t:
        return False
    if _looks_non_english(t):
        return False

    en_words = _EN_WORD_RE.findall(t)
    if len(en_words) < int(min_words):
        return False

    en_alpha = len(_EN_ALPHA_RE.findall(t))
    uni_alpha = len(_UNICODE_ALPHA_RE.findall(t))
    if uni_alpha <= 0:
        return False
    if (en_alpha / float(uni_alpha)) < float(min_alpha_ratio):
        return False
    return True


def _normalize_tex_math_text(text: str) -> str:
    t = str(text or "")
    if not t:
        return ""

    # Repair broken backslash-command splits across lines.
    t = re.sub(r"\\\s*\n\s*([A-Za-z]+)", r"\\\1", t)
    # Collapse duplicated escape backslashes before commands/delimiters.
    t = re.sub(r"\\\\(?=[\[\]\(\){}])", r"\\", t)
    t = re.sub(r"\\\\(?=[A-Za-z])", r"\\", t)

    rep = {
        "≤": r"\le",
        "≥": r"\ge",
        "≠": r"\ne",
        "≈": r"\approx",
        "∈": r"\in",
        "∉": r"\notin",
        "⊂": r"\subset",
        "⊆": r"\subseteq",
        "⊃": r"\supset",
        "⊇": r"\supseteq",
        "→": r"\to",
        "⇒": r"\Rightarrow",
        "↦": r"\mapsto",
        "×": r"\times",
        "·": r"\cdot",
        "∞": r"\infty",
        "∇": r"\nabla",
        "ℝ": r"\mathbf{R}",
        "ℕ": r"\mathbf{N}",
        "ℤ": r"\mathbf{Z}",
        "ℚ": r"\mathbf{Q}",
        "ℂ": r"\mathbf{C}",
    }
    for k, v in rep.items():
        t = t.replace(k, v)

    t = t.replace(r"\mathbb{R}", r"\mathbf{R}")
    t = t.replace(r"\mathbb{N}", r"\mathbf{N}")
    t = t.replace(r"\mathbb{Z}", r"\mathbf{Z}")
    t = t.replace(r"\mathbb{Q}", r"\mathbf{Q}")
    t = t.replace(r"\mathbb{C}", r"\mathbf{C}")

    t = re.sub(r"(?<!\\)\bmathbb([RNCZQ])\b", r"\\mathbf{\1}", t)
    t = re.sub(r"(?<!\\)\bmathbf([RNCZQ])\b", r"\\mathbf{\1}", t)
    t = re.sub(r"(?<!\\)\bmathbb\{([RNCZQ])\}", r"\\mathbf{\1}", t)
    t = re.sub(r"(?<!\\)\bmathbf\{([RNCZQ])\}", r"\\mathbf{\1}", t)

    greek = ["alpha", "beta", "gamma", "lambda", "mu", "theta", "sigma", "phi", "psi", "omega", "nabla"]
    for g in greek:
        t = re.sub(rf"(?<!\\)\b{g}\b", rf"\\{g}", t)

    # Common OCR drop: missing leading "n" in nabla.
    t = re.sub(r"(?<!\\)\babla\b", r"\\nabla", t)

    t = re.sub(r"(?<=\w)times(?=\w)", r"\\times", t)
    t = re.sub(r"(?<!\\)\bcdot\b", r"\\cdot", t)

    # Normalize dollar-delimited math.
    t = re.sub(r"(?s)\$\$\s*(.*?)\s*\$\$", r"\\[\1\\]", t)
    t = re.sub(r"(?s)(?<!\$)\$(?!\$)\s*(.+?)\s*(?<!\$)\$(?!\$)", r"\\(\1\\)", t)

    # Convert one-line display math to inline when embedded in prose.
    t = re.sub(r"\\\[\s*([^\n]+?)\s*\\\]", r"\\(\1\\)", t)

    t = re.sub(r"\\\(\s*\n\s*", r"\\(", t)
    t = re.sub(r"\s*\n\s*\\\)", r"\\)", t)
    t = t.replace(r"\(\n", r"\(")
    t = t.replace(r"\n\)", r"\)")

    t = re.sub(r"\\n(?=[A-Z])", "\n", t)
    t = "\n".join(re.sub(r"[ \t]+", " ", line).rstrip() for line in t.splitlines())
    return t.strip()


def _is_task_sentence(s: str) -> bool:
    lead = s.strip()
    if not lead:
        return False
    lead = re.sub(r"^(Equivalently|Also|Then|Next|Moreover|Furthermore|Hence|Thus|Finally)\s*,?\s+", "", lead, flags=re.IGNORECASE)
    return bool(re.match(r"^(Prove|Show|Find|Compute|Derive|Determine|Verify|Establish|Evaluate|Minimize|Maximize)\b", lead, flags=re.IGNORECASE))


def _split_inline_marked_questions(text: str) -> Optional[str]:
    t = str(text or "").strip()
    if not t:
        return None

    # Only treat explicit sub-question markers at clause boundaries.
    marks = list(re.finditer(r"(?m)(^|[\n;]\s*)\(([ivxIVX]+|[a-zA-Z])\)\s+", t))
    if len(marks) < 2:
        return None

    pieces: List[str] = []
    for i, m in enumerate(marks):
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(t)
        chunk = t[start:end].strip(" \n;")
        if chunk:
            pieces.append(chunk)
    if len(pieces) < 2:
        return None
    return "\n".join(f"{i}. {p}" for i, p in enumerate(pieces, start=1))
def _number_multi_questions(text: str) -> str:
    t = str(text or "").strip()
    if not t:
        return ""

    if re.search(r"(?m)^\s*\d+\.\s+\S", t):
        return t

    marked = _split_inline_marked_questions(t)
    if marked:
        return marked

    dm = re.match(r"(?is)^(.*?)(Determine|Decide)\s+whether\s+(.+?)\.?$", t)
    if dm:
        head = dm.group(1).strip()
        tail = dm.group(3).strip()
        tail = re.sub(r"\s+and\s+", ", ", tail)
        parts = [x.strip(" ;") for x in tail.split(",") if x.strip()]
        if len(parts) >= 2:
            body = "\n".join(f"{i}. {part}." for i, part in enumerate(parts, start=1))
            prefix = (head + " Determine whether:").strip()
            return (prefix + "\n" + body).strip()

    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    if len(sents) < 2:
        return t

    first_task = -1
    for i, s in enumerate(sents):
        if _is_task_sentence(s):
            first_task = i
            break
    if first_task < 0:
        return t

    tasks: List[str] = []
    for s in sents[first_task:]:
        if _is_task_sentence(s):
            tasks.append(s)
        elif tasks:
            tasks[-1] = (tasks[-1] + " " + s).strip()
    if len(tasks) < 2:
        return t

    prefix = " ".join(sents[:first_task]).strip()
    body = "\n".join(f"{i}. {q}" for i, q in enumerate(tasks, start=1))
    return (prefix + "\n" + body).strip() if prefix else body


def _math_statement_sane(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return False

    # Basic delimiter sanity checks.
    if t.count(r"\(") != t.count(r"\)"):
        return False
    if t.count(r"\[") != t.count(r"\]"):
        return False

    # Reject clearly broken numbered items like "1. \le f" or "1. - ...".
    bad_item = re.compile(r"(?m)^\s*\d+\.\s*(?:[-,:;]|\\(?:le|ge|ne|in|to|times|cdot|subset|subseteq)\b)")
    if bad_item.search(t):
        return False

    # Reject residual Unicode math symbols; require standard LaTeX commands instead.
    if re.search(r"[≤≥≠≈∈∉⊂⊆⊃⊇→⇒↦×·∞∇ℝℕℤℚℂ]", t):
        return False

    for m in re.finditer(r"(?m)^\s*\d+\.\s*(.+)$", t):
        item = m.group(1).strip()
        if len(item) < 6 or not re.search(r"[A-Za-z]", item):
            return False

    return True


def _rewrite_field_via_llm(
    *,
    client: OpenAI,
    model: str,
    prompt: str,
    out_keys: List[str],
    max_tokens: int,
    retries: int,
    cache_dir: Optional[Path],
    cache_enabled: bool,
    min_english_words: int,
    min_english_alpha_ratio: float,
) -> Tuple[str, str]:
    reasons: List[str] = []
    attempts = max(1, int(retries))
    for i in range(1, attempts + 1):
        raw_i = _call_cached(
            client,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
        )
        text_i = _extract_json_string_by_keys(raw_i, out_keys)
        if not text_i:
            reasons.append(f"attempt{i}:empty_or_unparseable")
            continue
        if not _english_quality_ok(
            text_i,
            min_words=int(min_english_words),
            min_alpha_ratio=float(min_english_alpha_ratio),
        ):
            reasons.append(f"attempt{i}:non_english_or_low_quality")
            continue
        if not _math_statement_sane(_normalize_tex_math_text(text_i)):
            reasons.append(f"attempt{i}:tex_or_structure_invalid")
            continue
        return text_i, "ok"
    return "", "; ".join(reasons) if reasons else "empty_or_unparseable"


def finalize_problem_three_step(
    *,
    problem_standardized_math: str,
    client: Optional[OpenAI],
    use_llm: bool,
    model: str,
    max_tokens: int,
    llm_retries: int,
    cache_dir: Optional[Path],
    cache_enabled: bool,
    min_english_words: int,
    min_english_alpha_ratio: float,
    progress_prefix: str = "",
) -> Tuple[str, str]:
    base = _normalize_tex_math_text(str(problem_standardized_math or "").strip())
    if not base:
        if progress_prefix:
            print(f"{progress_prefix} empty standardized text; skip", file=sys.stderr)
        return "", "empty_problem_standardized_math"
    if (not use_llm) or client is None:
        if progress_prefix:
            print(f"{progress_prefix} llm unavailable; reuse standardized", file=sys.stderr)
        return base, "llm_disabled_or_missing_key"

    if progress_prefix:
        print(f"{progress_prefix} step1/2 COMPLETE start", file=sys.stderr)
    p1, n1 = _rewrite_field_via_llm(
        client=client,
        model=model,
        prompt=make_problem_complete_prompt(base),
        out_keys=["problem_complete"],
        max_tokens=max_tokens,
        retries=llm_retries,
        cache_dir=cache_dir,
        cache_enabled=cache_enabled,
        min_english_words=min_english_words,
        min_english_alpha_ratio=min_english_alpha_ratio,
    )
    if not p1:
        if progress_prefix:
            print(f"{progress_prefix} step1/2 COMPLETE fallback -> standardized ({n1})", file=sys.stderr)
        return base, "complete_fallback:" + n1
    p1 = _normalize_tex_math_text(p1)

    if progress_prefix:
        print(f"{progress_prefix} step2/2 CONCISE start", file=sys.stderr)
    p2, n2 = _rewrite_field_via_llm(
        client=client,
        model=model,
        prompt=make_problem_concise_prompt(p1),
        out_keys=["problem_concise"],
        max_tokens=max_tokens,
        retries=llm_retries,
        cache_dir=cache_dir,
        cache_enabled=cache_enabled,
        min_english_words=min_english_words,
        min_english_alpha_ratio=min_english_alpha_ratio,
    )
    if not p2:
        if progress_prefix:
            print(f"{progress_prefix} step2/2 CONCISE fallback -> complete ({n2})", file=sys.stderr)
        return p1, "concise_fallback:" + n2
    p2 = _number_multi_questions(_normalize_tex_math_text(p2))
    if not _math_statement_sane(p2):
        if progress_prefix:
            print(f"{progress_prefix} step2/2 output invalid -> complete", file=sys.stderr)
        return p1, "concise_invalid_fallback_complete"

    if progress_prefix:
        print(f"{progress_prefix} done", file=sys.stderr)
    # Third step removed by design: use concise text as final output.
    return p2, "ok"


def _fallback_problem(record: Dict[str, Any]) -> Tuple[str, str, str]:
    pwc = str(record.get("problem_with_context") or "").strip()
    prob = str(record.get("problem") or "").strip()
    if pwc:
        return pwc, "fallback_context", "llm_empty_or_unavailable"
    if prob:
        return prob, "fallback_original", "llm_empty_or_unavailable"
    return "", "failed", "empty_problem_and_context"


def naturalize_one(
    record: Dict[str, Any],
    *,
    client: Optional[OpenAI],
    model: str,
    max_tokens: int,
    prompt_version: str,
    cache_dir: Optional[Path],
    cache_enabled: bool,
    use_llm: bool,
    llm_retries: int,
    min_english_words: int,
    min_english_alpha_ratio: float,
    enable_problem_finally: bool,
    final_max_tokens: int,
    final_llm_retries: int,
    reuse_existing_standardized: bool,
    progress_prefix: str = "",
) -> Dict[str, Any]:
    out = dict(record)

    def _finalize_out_fields(row: Dict[str, Any]) -> Dict[str, Any]:
        row["problem_with_context"] = _normalize_tex_math_text(str(row.get("problem_with_context") or ""))
        if "problem_standardized_math" in row:
            row["problem_standardized_math"] = _normalize_tex_math_text(str(row.get("problem_standardized_math") or ""))
        if "problem_finally" in row:
            row["problem_finally"] = _normalize_tex_math_text(str(row.get("problem_finally") or ""))
        return row

    if progress_prefix:
        print(f"{progress_prefix} standardized stage start", file=sys.stderr)
    existing_standardized = str(out.get("problem_standardized_math") or "").strip()

    if reuse_existing_standardized and existing_standardized:
        if progress_prefix:
            print(f"{progress_prefix} standardized reused", file=sys.stderr)
        out["naturalize_status"] = "skipped"
        out["naturalize_prompt_version"] = prompt_version
        out["naturalize_notes"] = "reuse_existing_problem_standardized_math"
        if enable_problem_finally:
            pf, pf_note = finalize_problem_three_step(
                problem_standardized_math=existing_standardized,
                client=client,
                use_llm=use_llm,
                model=model,
                max_tokens=max(200, int(final_max_tokens)),
                llm_retries=max(1, int(final_llm_retries)),
                cache_dir=cache_dir,
                cache_enabled=cache_enabled,
                min_english_words=max(1, int(min_english_words)),
                min_english_alpha_ratio=max(0.0, min(1.0, float(min_english_alpha_ratio))),
                progress_prefix=progress_prefix,
            )
            out["problem_finally"] = _normalize_tex_math_text(pf)
            if pf_note != "ok":
                out["naturalize_notes"] = f"{out['naturalize_notes']}; problem_finally:{pf_note}"
        return _finalize_out_fields(out)

    if not use_llm or client is None:
        if progress_prefix:
            print(f"{progress_prefix} standardized fallback (llm unavailable)", file=sys.stderr)
        text, status, notes = _fallback_problem(record)
        out["problem_standardized_math"] = _normalize_tex_math_text(text)
        if enable_problem_finally:
            out["problem_finally"] = _normalize_tex_math_text(text)
        out["naturalize_status"] = status
        out["naturalize_prompt_version"] = prompt_version
        out["naturalize_notes"] = "llm_disabled_or_missing_key; " + notes
        return _finalize_out_fields(out)

    payload = build_naturalize_input(record)
    prompt = make_prompt(payload)

    try:
        attempts = max(1, int(llm_retries))
        reasons: List[str] = []
        standardized = ""

        for i in range(1, attempts + 1):
            prompt_i = prompt if i == 1 else make_retry_prompt(payload)
            raw_i = _call_cached(
                client,
                model=model,
                prompt=prompt_i,
                max_tokens=max_tokens,
                cache_dir=cache_dir,
                cache_enabled=cache_enabled,
            )
            standardized = _extract_standardized_text(raw_i)
            if not standardized:
                reasons.append(f"attempt{i}:empty_or_unparseable")
                continue
            if not _english_quality_ok(
                standardized,
                min_words=int(min_english_words),
                min_alpha_ratio=float(min_english_alpha_ratio),
            ):
                reasons.append(f"attempt{i}:non_english_or_low_quality")
                continue
            if not _math_statement_sane(_normalize_tex_math_text(standardized)):
                reasons.append(f"attempt{i}:tex_or_structure_invalid")
                continue
            break

        if not standardized or not _english_quality_ok(
            standardized,
            min_words=int(min_english_words),
            min_alpha_ratio=float(min_english_alpha_ratio),
        ):
            if progress_prefix:
                print(f"{progress_prefix} standardized fallback (quality/parse)", file=sys.stderr)
            text, status, notes = _fallback_problem(record)
            out["problem_standardized_math"] = _normalize_tex_math_text(text)
            if enable_problem_finally:
                out["problem_finally"] = _normalize_tex_math_text(text)
            out["naturalize_status"] = status
            out["naturalize_prompt_version"] = prompt_version
            out["naturalize_notes"] = "; ".join(reasons + [notes]) if reasons else notes
            return _finalize_out_fields(out)

        standardized = _normalize_tex_math_text(standardized)
        out["problem_standardized_math"] = standardized
        if progress_prefix:
            print(f"{progress_prefix} standardized ok", file=sys.stderr)
        if enable_problem_finally:
            pf, pf_note = finalize_problem_three_step(
                problem_standardized_math=standardized,
                client=client,
                use_llm=use_llm,
                model=model,
                max_tokens=max(200, int(final_max_tokens)),
                llm_retries=max(1, int(final_llm_retries)),
                cache_dir=cache_dir,
                cache_enabled=cache_enabled,
                min_english_words=max(1, int(min_english_words)),
                min_english_alpha_ratio=max(0.0, min(1.0, float(min_english_alpha_ratio))),
                progress_prefix=progress_prefix,
            )
            out["problem_finally"] = _normalize_tex_math_text(pf)
        out["problem_with_context"] = _normalize_tex_math_text(str(out.get("problem_with_context") or ""))
        out["naturalize_status"] = "ok"
        out["naturalize_prompt_version"] = prompt_version
        out["naturalize_notes"] = ""
        if enable_problem_finally and pf_note != "ok":
            out["naturalize_notes"] = f"problem_finally:{pf_note}"
        return _finalize_out_fields(out)
    except Exception as e:
        text, status, notes = _fallback_problem(record)
        out["problem_standardized_math"] = _normalize_tex_math_text(text)
        if enable_problem_finally:
            out["problem_finally"] = _normalize_tex_math_text(text)
        out["naturalize_status"] = status
        out["naturalize_prompt_version"] = prompt_version
        out["naturalize_notes"] = f"llm_error={e.__class__.__name__}; {notes}"
        return _finalize_out_fields(out)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Step-2 naturalize step-1 JSON records.")
    ap.add_argument("in_json", type=str, help="Input step-1 JSON file")
    ap.add_argument("out_json", nargs="?", default="", help="Output step-2 JSON file (optional)")
    ap.add_argument(
        "--out-dir",
        type=str,
        default="output_json_naturalized",
        help="Default output root dir when out_json is omitted",
    )
    ap.add_argument("--model", type=str, default="", help="LLM model override (default: config.model)")
    ap.add_argument("--max-tokens", type=int, default=900, help="LLM max tokens per item")
    ap.add_argument("--max-items", type=int, default=0, help="Max rows to send to LLM (0 = all)")
    ap.add_argument("--prompt-version", type=str, default=PROMPT_VERSION_DEFAULT, help="Prompt version tag")
    ap.add_argument("--cache-dir", type=str, default="", help="Cache dir (default: cache/naturalize)")
    ap.add_argument("--no-cache", action="store_true", help="Disable LLM cache")
    ap.add_argument("--disable-llm", action="store_true", help="Do not call LLM; fallback only")
    ap.add_argument("--force", action="store_true", help="Rewrite rows even if problem_standardized_math exists")
    ap.add_argument("--llm-retries", type=int, default=3, help="Max LLM attempts per row")
    ap.add_argument("--min-english-words", type=int, default=6, help="Min English word count for accepted output")
    ap.add_argument(
        "--min-english-alpha-ratio",
        type=float,
        default=0.6,
        help="Min ratio of ASCII English letters among all alphabetic chars",
    )
    ap.add_argument("--stats-out", type=str, default="", help="Optional path to write per-file stats JSON")
    ap.add_argument(
        "--disable-problem-finally",
        action="store_true",
        help="Disable two-step refinement and do not output problem_finally",
    )
    ap.add_argument(
        "--final-max-tokens",
        type=int,
        default=900,
        help="Max tokens per step for problem_finally two-step refinement",
    )
    ap.add_argument(
        "--final-llm-retries",
        type=int,
        default=3,
        help="Max retries per step for problem_finally two-step refinement",
    )
    ap.add_argument(
        "--no-cleanup-cache-on-exit",
        action="store_true",
        help="Keep naturalize cache directory after run",
    )
    return ap.parse_args()


def _derive_out_path(in_path: Path, out_json: str, out_dir: str) -> Path:
    if out_json:
        return Path(out_json).expanduser().resolve()

    project_root = find_config_json().parent
    base_step1 = project_root / "output_json"
    out_root = project_root / str(out_dir or "output_json_naturalized")

    try:
        rel = in_path.relative_to(base_step1)
        return (out_root / rel).resolve()
    except Exception:
        return in_path.with_name(in_path.stem + ".naturalized" + in_path.suffix).resolve()


def _strip_naturalize_meta(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        r.pop("naturalize_status", None)
        r.pop("naturalize_prompt_version", None)
        r.pop("naturalize_notes", None)
        # Final output should stay compact/readable: remove dependency bundle and transient dependency fields.
        r.pop("dependency", None)
        for k in [
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
        ]:
            r.pop(k, None)
        out.append(r)
    return out


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_json).expanduser().resolve()
    out_path = _derive_out_path(in_path, str(args.out_json or ""), str(args.out_dir or ""))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_raw = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(rows_raw, list):
        raise ValueError("Input JSON must be a list of records.")
    rows: List[Dict[str, Any]] = [r if isinstance(r, dict) else {} for r in rows_raw]

    cfg = {}
    try:
        cfg = load_config()
    except Exception:
        cfg = {}
    api_key = str(cfg.get("api_key") or os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = str(cfg.get("base_url") or "").strip()
    model = str(args.model or cfg.get("model") or "gpt-5-mini").strip()
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None
    cache_enabled = not bool(args.no_cache)
    cleanup_cache = not bool(args.no_cleanup_cache_on_exit)
    effective_cache_dir = cache_dir or (Path.cwd() / "cache" / "naturalize")

    use_llm = (not bool(args.disable_llm)) and bool(api_key)
    client: Optional[OpenAI] = None
    if use_llm:
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=180)
        else:
            client = OpenAI(api_key=api_key, timeout=180)

    budget = max(0, int(args.max_items))
    touched = 0
    out_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    ok_cnt = 0
    fb_cnt = 0
    skip_cnt = 0
    fail_cnt = 0

    pbar = tqdm(rows, total=len(rows), desc="Naturalize rows", unit="row")
    enable_problem_finally = not bool(args.disable_problem_finally)
    for idx, row in enumerate(pbar, start=1):
        has_prev = str(row.get("problem_standardized_math") or "").strip()
        has_final = str(row.get("problem_finally") or "").strip()
        can_skip = bool(has_prev) and (not enable_problem_finally or bool(has_final))
        if can_skip and not args.force:
            kept = dict(row)
            out_rows.append(kept)
            st = str(kept.get("naturalize_status") or "skipped")
            if st == "ok":
                ok_cnt += 1
            elif st.startswith("fallback"):
                fb_cnt += 1
            elif st == "failed":
                fail_cnt += 1
            else:
                skip_cnt += 1
            pbar.set_postfix({
                "llm": touched,
                "ok": ok_cnt,
                "fb": fb_cnt,
                "skip": skip_cnt,
                "fail": fail_cnt,
            })
            continue
        can_use_llm = use_llm and (budget == 0 or touched < budget)
        if can_use_llm:
            touched += 1
        label = str(row.get("source_idx") or row.get("index") or f"row{idx}")
        progress_prefix = f"[problem_finally {idx}/{len(rows)} {label}]"
        new_row = naturalize_one(
            row,
            client=client,
            model=model,
            max_tokens=max(200, int(args.max_tokens)),
            prompt_version=str(args.prompt_version or PROMPT_VERSION_DEFAULT),
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
            use_llm=can_use_llm,
            llm_retries=max(1, int(args.llm_retries)),
            min_english_words=max(1, int(args.min_english_words)),
            min_english_alpha_ratio=max(0.0, min(1.0, float(args.min_english_alpha_ratio))),
            enable_problem_finally=enable_problem_finally,
            final_max_tokens=max(200, int(args.final_max_tokens)),
            final_llm_retries=max(1, int(args.final_llm_retries)),
            reuse_existing_standardized=bool(has_prev and not args.force),
            progress_prefix=progress_prefix,
        )
        out_rows.append(new_row)
        st = str(new_row.get("naturalize_status") or "")
        if st == "ok":
            ok_cnt += 1
        elif st.startswith("fallback"):
            fb_cnt += 1
        elif st == "failed":
            fail_cnt += 1
        else:
            skip_cnt += 1
        pbar.set_postfix({
            "llm": touched,
            "ok": ok_cnt,
            "fb": fb_cnt,
            "skip": skip_cnt,
            "fail": fail_cnt,
        })

    pbar.close()

    output_rows = _strip_naturalize_meta(out_rows)
    out_path.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    dt = time.time() - t0
    ok = sum(1 for r in out_rows if str(r.get("naturalize_status") or "") == "ok")
    fb = sum(1 for r in out_rows if str(r.get("naturalize_status") or "").startswith("fallback"))
    skip = sum(1 for r in out_rows if str(r.get("naturalize_status") or "") == "skipped")
    fail = sum(1 for r in out_rows if str(r.get("naturalize_status") or "") == "failed")

    if args.stats_out:
        stats_path = Path(str(args.stats_out)).expanduser().resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_obj = {
            "in_json": str(in_path),
            "out_json": str(out_path),
            "rows": int(len(out_rows)),
            "llm_touched": int(touched),
            "ok": int(ok),
            "fallback": int(fb),
            "skipped": int(skip),
            "failed": int(fail),
            "seconds": float(round(dt, 4)),
        }
        stats_path.write_text(json.dumps(stats_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "DONE: "
        f"{out_path} (rows={len(out_rows)}, llm_touched={touched}, ok={ok}, "
        f"fallback={fb}, skipped={skip}, failed={fail}, sec={dt:.2f})"
    )

    if cleanup_cache:
        try:
            if effective_cache_dir.exists():
                shutil.rmtree(effective_cache_dir)
                print(f"CACHE CLEANED: {effective_cache_dir}")
        except Exception as e:
            print(f"CACHE CLEAN FAILED: {effective_cache_dir} ({e.__class__.__name__})")


if __name__ == "__main__":
    main()
