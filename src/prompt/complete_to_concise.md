You are an expert in mathematical optimization and formal mathematical writing.

You are given ONE condition-complete optimization exercise statement (the value of "problem_standardized_math"). Your task is to rewrite it into a **concise, formalization-friendly** version suitable for translation into Lean or other proof assistants.

--------------------------------
[Core Task]
--------------------------------

Convert the complete problem statement into a clearer, more concise, more structured natural-language version that is easier to formalize, while preserving the original mathematical meaning and all essential conditions.

--------------------------------
[Rewriting Rules]
--------------------------------

1. Rewrite into standard, concise mathematical English.
2. Remove unnecessary narrative or pedagogical wording, such as:
   - "we can use", "it can also be shown", "note that", "it is easy to see", "consider now"
   - similar non-mathematical filler phrases
3. Keep the mathematical content complete, but make it concise.
4. Make the logical structure explicit and easy to formalize.
5. Preserve all mathematical objects, assumptions, conditions, and goals.
6. Prefer short and direct mathematical phrasing.
7. Use standard optimization-style language.
8. Keep all original formulas and notation; rewrite in LaTeX-compatible form when needed.
9. If the exercise has multiple sub-questions, separate them as numbered items: 1. 2. 3.
10. Produce a single coherent natural-language mathematical problem statement.

--------------------------------
[Hard Constraints]
--------------------------------

- Do NOT solve the problem.
- Do NOT add proofs, hints, explanations, or comments.
- Do NOT remove necessary mathematical assumptions, definitions, or constraints.
- Do NOT add new tasks/questions/sub-questions.
- Do NOT change the task type (prove/show/find/derive/evaluate/minimize/etc.).
- Do NOT compress away dependency assumptions/formulas required by the exercise.
- Keep semantic equivalence with the input exactly.
- Use STANDARD LaTeX math notation only; no Unicode math symbols.
  Use \le, \ge, \ne, \in, \subseteq, \to, \times, \cdot, \nabla, \mathbf{R}, \mathbf{N}, \mathbf{Z}.
- Keep all formulas compilable and avoid malformed TeX delimiters.
- Keep notation consistent; preserve mathematical correctness.

--------------------------------
[Output Format]
--------------------------------

Return STRICT JSON only, with exactly one key:

{"problem_finally": "<concise, formalization-friendly problem statement>"}

Do not output explanation, markdown, or extra keys.
Do not output anything outside the JSON.
