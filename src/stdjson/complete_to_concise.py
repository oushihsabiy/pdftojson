#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Rewrite the 'problem' field of exercise JSON objects into a clearer,
more concise form suitable for later Lean formalization.
Uses the complete_to_concise prompt.

Usage (standalone):
    python complete_to_concise.py input.json output.json [--prompt PATH] [--model NAME]

When invoked from the pipeline (main.py), the positional arguments are filled
automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI


DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT_SECONDS = 180.0
CHAT_FORCE_STREAM: Optional[bool] = None

DEFAULT_FALLBACK_PROMPT = """
You are an expert in optimization exercise normalization.

Task:
- Input is one exercise JSON object.
- Rewrite only the "problem" field to make it clearer, more concise, and easier to translate into Lean.
- Preserve the original mathematical intent, question type, and difficulty.
- Keep all non-"problem" fields exactly unchanged.

Rules:
- Do not solve the problem.
- Do not add explanations.
- Return exactly one JSON object and nothing else.
- Keep keys and key order unchanged.
""".strip()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def find_config_json() -> Path:
    cwd_path = Path.cwd() / "config.json"
    if cwd_path.exists():
        return cwd_path.resolve()

    here = Path(__file__).resolve().parent
    for parent in [here] + list(here.parents):
        candidate = parent / "config.json"
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "config.json not found (checked CWD and script parents).")


def load_config() -> Dict[str, Any]:
    config_path = find_config_json()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a JSON object.")
    return data


def require_str(config: Dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise KeyError(
            f"Missing/invalid '{key}' in config.json (must be non-empty string)")
    return value.strip()


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def find_prompt_path(prompt_arg: Optional[str]) -> Path:
    if prompt_arg:
        prompt_path = Path(prompt_arg).expanduser().resolve()
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        return prompt_path

    default_path = Path(__file__).resolve(
    ).parents[1] / "prompt" / "complete_to_concise.md"
    if not default_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {default_path}")
    return default_path


def load_prompt(prompt_arg: Optional[str]) -> str:
    prompt_path = find_prompt_path(prompt_arg)
    text = prompt_path.read_text(encoding="utf-8").strip()
    if text:
        return text

    print(
        f"[warn] Prompt file is empty, using built-in fallback prompt: {prompt_path}",
        file=sys.stderr,
    )
    return DEFAULT_FALLBACK_PROMPT


# ---------------------------------------------------------------------------
# Exercise detection & iteration
# ---------------------------------------------------------------------------

def is_exercise_object(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    if "problem" not in node:
        return False
    marker_keys = {"proof", "direct_answer",
                   "source_idx", "source", "题目类型", "预估难度"}
    return any(key in node for key in marker_keys)


def iter_exercise_objects(node: Any) -> Iterable[Dict[str, Any]]:
    if is_exercise_object(node):
        yield node
        return

    if isinstance(node, list):
        for item in node:
            yield from iter_exercise_objects(item)
        return

    if isinstance(node, dict):
        for item in node.values():
            yield from iter_exercise_objects(item)


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# LLM chat helpers (stream-aware)
# ---------------------------------------------------------------------------

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
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content") or ""
                    else:
                        text = getattr(item, "text", "") or getattr(
                            item, "content", "") or ""
                    if isinstance(text, str) and text:
                        parts.append(text)
    finally:
        close_fn = getattr(stream_obj, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
    return "".join(parts)


def chat_completion_text(
        client: OpenAI,
        *,
        model: str,
        prompt: str,
        max_tokens: int,
) -> str:
    global CHAT_FORCE_STREAM

    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
    )

    try:
        kwargs["response_format"] = {"type": "json_object"}
    except Exception:
        pass

    if CHAT_FORCE_STREAM is True:
        stream_obj = client.chat.completions.create(stream=True, **kwargs)
        return _collect_stream_text(stream_obj).strip()

    try:
        response = client.chat.completions.create(**kwargs)
        return (response.choices[0].message.content or "").strip()
    except Exception as err:
        if _is_stream_required_error(err):
            CHAT_FORCE_STREAM = True
            stream_obj = client.chat.completions.create(stream=True, **kwargs)
            return _collect_stream_text(stream_obj).strip()
        raise


# ---------------------------------------------------------------------------
# JSON extraction from model output
# ---------------------------------------------------------------------------

def extract_json_value(text: str) -> Any:
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("Model returned empty output.")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    code_fence_start = stripped.find("```")
    if code_fence_start >= 0:
        code_fence_end = stripped.rfind("```")
        if code_fence_end > code_fence_start:
            fenced = stripped[code_fence_start + 3:code_fence_end].strip()
            newline = fenced.find("\n")
            if newline >= 0 and fenced[:newline].strip().lower() == "json":
                fenced = fenced[newline + 1:].strip()
            try:
                return json.loads(fenced)
            except json.JSONDecodeError:
                pass

    object_start = stripped.find("{")
    object_end = stripped.rfind("}")
    if object_start >= 0 and object_end > object_start:
        snippet = stripped[object_start:object_end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    array_start = stripped.find("[")
    array_end = stripped.rfind("]")
    if array_start >= 0 and array_end > array_start:
        snippet = stripped[array_start:array_end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    raise ValueError("Model output is not valid JSON.")


# ---------------------------------------------------------------------------
# Prompt construction & validation
# ---------------------------------------------------------------------------

def build_single_object_prompt(base_prompt: str, exercise: Dict[str, Any], feedback: str = "") -> str:
    object_json = json.dumps(exercise, ensure_ascii=False, indent=2)
    extra = ""
    if feedback:
        extra = (
            "\n\nThe previous output was invalid. Fix it strictly according to this validation feedback:\n"
            f"{feedback}\n"
        )

    return (
        f"{base_prompt}"
        "\n\nProcess exactly one exercise object."
        " Return exactly one JSON object and nothing else."
        " Keep every field other than \"problem\" exactly unchanged."
        f"{extra}"
        "\n\nInput JSON object:\n"
        f"{object_json}"
    )


def validate_candidate(original: Dict[str, Any], candidate: Any) -> str:
    if isinstance(candidate, list):
        if len(candidate) != 1:
            return "Output must be a single JSON object, not an array with multiple items."
        candidate = candidate[0]

    if not isinstance(candidate, dict):
        return "Output must decode to a JSON object."

    if list(candidate.keys()) != list(original.keys()):
        return "The output object must keep exactly the same keys in exactly the same order."

    for key, value in original.items():
        if key == "problem":
            continue
        if candidate.get(key) != value:
            return f"Field '{key}' was modified, but only 'problem' may change."

    problem_value = candidate.get("problem")
    if not isinstance(problem_value, str):
        return "Field 'problem' must remain a string."

    return ""


# ---------------------------------------------------------------------------
# Core: rewrite one exercise's problem field
# ---------------------------------------------------------------------------

def concise_rewrite_problem(
        client: OpenAI,
        *,
        model: str,
        base_prompt: str,
        exercise: Dict[str, Any],
        max_tokens: int,
        max_attempts: int,
) -> str:
    feedback = ""
    last_error = ""

    for _ in range(max_attempts):
        prompt = build_single_object_prompt(base_prompt, exercise, feedback)
        response_text = chat_completion_text(
            client,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        try:
            candidate = extract_json_value(response_text)
        except Exception as err:
            last_error = str(err)
            feedback = f"Output parsing failed: {err}"
            continue

        validation_error = validate_candidate(exercise, candidate)
        if validation_error:
            last_error = validation_error
            feedback = validation_error
            continue

        if isinstance(candidate, list):
            candidate = candidate[0]
        return candidate["problem"]

    raise RuntimeError(
        f"Failed to obtain a valid concise problem after {max_attempts} attempts: {last_error}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite the 'problem' field into a clearer, more concise form for later Lean translation.",
    )
    parser.add_argument("input_json", help="Path to the input JSON file.")
    parser.add_argument("output_json", help="Path to the output JSON file.")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Path to the prompt markdown file. Defaults to src/prompt/complete_to_concise.md.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model name from config.json.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum completion tokens per object (default: {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=8,
        help="Maximum validation/retry attempts per object (default: 8).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Client timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    prompt_text = load_prompt(args.prompt)

    api_key = require_str(config, "api_key")
    base_url = require_str(config, "base_url")
    model = args.model or require_str(config, "model")

    input_path = Path(args.input_json).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()

    data = read_json(input_path)
    exercises = list(iter_exercise_objects(data))
    if not exercises:
        raise ValueError(
            "No exercise objects with a 'problem' field were found in the input JSON.")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout)

    total = len(exercises)
    failed_labels: List[str] = []
    for index, exercise in enumerate(exercises, start=1):
        label = exercise.get("source_idx") or exercise.get("index") or index
        print(f"[{index}/{total}] concise-rewriting problem for {label}",
              file=sys.stderr)
        try:
            new_problem = concise_rewrite_problem(
                client,
                model=model,
                base_prompt=prompt_text,
                exercise=exercise,
                max_tokens=args.max_tokens,
                max_attempts=args.max_attempts,
            )
            exercise["problem"] = new_problem
        except Exception as err:
            failed_labels.append(str(label))
            print(
                f"[warn] concise-rewriting failed for {label}; keep original problem. reason={err}",
                file=sys.stderr,
            )

    write_json(output_path, data)
    print(f"Wrote {total} concise-rewritten exercise(s) to {output_path}",
          file=sys.stderr)
    if failed_labels:
        print(
            f"[warn] {len(failed_labels)} exercise(s) kept original problem: {', '.join(failed_labels)}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
