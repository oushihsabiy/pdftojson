You are given a JSON array of optimization problems. Each JSON object contains metadata fields such as "index", "source_idx", "source", "题目类型", "预估难度", "problem", "proof", and "direct_answer".

Your task is to rewrite only the value of the "problem" field by converting it into a labeled, Lean-friendly structured form. All other fields must be copied exactly without any change.

Goal:
Convert each problem statement into a compact structured block whose logical units are explicitly separated into:
- Definition
- Hypothesis
- Goal

Hard constraints:
1. Modify only the "problem" field.
2. Preserve every other field exactly as in the input.
3. Do not solve the problem.
4. Do not add proofs, hints, explanations, or comments.
5. Do not remove necessary mathematical assumptions, definitions, or constraints.
6. Output must remain valid JSON.
7. Keep the original JSON structure unchanged except for the rewritten "problem" field.
8. The rewritten "problem" field must remain a JSON string that can be rendered as LaTeX-aware text.
9. Preserve all mathematical notation in LaTeX-compatible syntax.
10. Do not use Unicode mathematical symbols such as "∈", "≤", "≥", "→", "ℝ", or "∇" in the rewritten "problem". Use LaTeX commands instead, such as `\in`, `\le`, `\ge`, `\to`, `\mathbf{R}`, `\nabla`.
11. Escape backslashes properly for JSON strings. For example, write `\\in`, `\\mathbf{R}`, `\\le`, `\\nabla` inside JSON output.

Labeling rules:
1. Rewrite the problem into short labeled units.
2. Use only the following labels:
   - Definition:
   - Hypothesis:
   - Goal:
3. Object introductions, domains, function declarations, set definitions, and notation explanations must be labeled as "Definition:".
4. Constraints, inequalities, convexity assumptions, differentiability assumptions, feasibility assumptions, and similar conditions must be labeled as "Hypothesis:".
5. The final task, such as "show", "prove", "determine whether", or "is ... convex?", must be labeled as "Goal:".
6. Any sentence that explains notation, such as "where ... means ...", "... denotes ...", "... stands for ...", or "we define ... by ...", must be rewritten as a separate "Definition:" line.
7. Split the statement into small information units whenever possible, so that each unit can later correspond to one Lean statement.
8. Keep the final goal concise.
9. Do not include narrative transitions.

Formatting rules for the rewritten "problem":
1. The rewritten "problem" must be a short structured natural-language block inside a single JSON string.
2. Each labeled unit must occupy its own rendered line.
3. Use LaTeX formatting that guarantees line breaks in rendered output, not just plain text newlines.
4. A recommended format is
   `\\[\\begin{aligned}
   &\\text{Definition: } ...\\\\
   &\\text{Hypothesis: } ...\\\\
   &\\text{Goal: } ...
   \\end{aligned}\\]`
5. Do not place two labeled units on the same rendered line.
6. Mathematical expressions inside each line should be written in LaTeX-compatible syntax.
7. Inline mathematics may be written with `$...$` if needed, but the entire block should remain suitable for LaTeX rendering.
8. Ensure the final JSON string is properly escaped.

Example style:
Input "problem":
"Let $a, b \\in \\mathbf{R}^n$ with $a < b$. Define
\\[
S = \\{x \\in \\mathbf{R}^n \\mid a \\le x \\le b\\}.
\\]
Prove that $S$ is convex."

Rewritten "problem":
"\\[\\begin{aligned}
&\\text{Definition: } a, b \\in \\mathbf{R}^n.\\\\
&\\text{Definition: } S = \\{x \\in \\mathbf{R}^n \\mid a \\le x \\le b\\}.\\\\
&\\text{Hypothesis: } a < b.\\\\
&\\text{Goal: prove that } S \\text{ is convex.}
\\end{aligned}\\]"

Now process the input JSON.
Remember:
- change only "problem"
- copy every other field exactly
- output only valid JSON
- do not add any extra text before or after the JSON
- all backslashes in LaTeX must be escaped properly for JSON