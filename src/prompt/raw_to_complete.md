You are an expert in mathematical optimization, mathematical writing, and exercise normalization.

You are given ONE mathematical exercise record extracted from a textbook. The record is a JSON object with these fields:
- "source_idx": exercise identifier (e.g., "Exercise 11.1")
- "source": source book/chapter identifier
- "original_problem": the raw exercise statement as extracted from the textbook
- "problem_with_context": the exercise statement with referenced formulas/theorems pre-inserted
- "context_references": list of reference lines (equations, theorems, algorithms) that appear in the exercise
- "equations": list of referenced equation objects, each with "tag", "display_tag", "equation_content"
- "contexts": list of referenced theorem/algorithm objects, each with "tag", "display_tag", "content"

Your goal is to rewrite the exercise into a **condition-complete, self-contained, standardized mathematical problem statement** in formal English, and return it as the value of the key "problem_standardized_math".

The exercises belong to the field of optimization, including but not limited to convex optimization, nonlinear programming, linear programming, duality, KKT conditions, subgradients, constrained optimization, variational analysis, and first-order methods.

--------------------------------
[Core Task]
--------------------------------

1. Start from "original_problem" as the primary source of the exercise statement.
2. Use "problem_with_context", "context_references", "equations", and "contexts" to resolve all references:
   - MUST replace pointers like "Theorem 11.8", "(11.46)", "Algorithm 11.5" with the actual mathematical content when it is available in the input.
   - If reference content is available, DO NOT write pointer-only phrases such as "assume the hypotheses of Theorem X.Y"; instead, spell out those hypotheses explicitly.
   - If a referenced formula/algorithm text is available, inline the concrete mathematical statement or procedure condition that is needed for the exercise.
   - If a reference is ambiguous or its content is not available in the input, keep the pointer text and do NOT invent details.
3. Identify implicit conditions that are omitted but truly necessary, and naturally incorporate them so the exercise becomes self-contained and well-posed.
4. Preserve the original mathematical intent, question type, and difficulty level.

--------------------------------
[Reference Integration Rules]
--------------------------------

Priority of information:
1. "original_problem" is the primary source of the exercise statement.
2. "context_references" and "problem_with_context" provide reference formulas/theorems/algorithms that MUST be integrated when clearly relevant.
3. "equations" contains formulas that must be preserved when they belong to the exercise.
4. "contexts" can help resolve nearby references, but you must not invent content from them.

When integrating references:
- Incorporate all clearly relevant referenced conditions/formulas into the standardized statement.
- COMPLETELY ELIMINATE all citation markers, equation numbers, and reference pointers from the output.
  Examples of forbidden residuals: "(11.35)", "Theorem 11.8", "Algorithm 11.5", "Eq. (3)", "equation (2.4)", "see (11.3)", "by Lemma 2.1", "from (A)", etc.
- Replace every such pointer with the actual mathematical content it refers to. If the content is available in the input, inline it explicitly.
- If a pointer's content is completely unavailable in the input, describe it in plain mathematical terms based on context, but never leave a bare numbered reference.
- After integration, the output must contain zero equation numbers, theorem numbers, algorithm numbers, or any other numbered citation labels.

--------------------------------
[Condition Completion Rules]
--------------------------------

You should add conditions that are clearly intended by the original exercise but not explicitly written, especially standard assumptions commonly omitted in optimization textbooks, such as:
- convexity of functions or sets
- differentiability, continuous differentiability, twice continuous differentiability
- continuity
- nonemptiness, closedness, compactness, convexity, boundedness, openness of sets
- symmetry and positive definiteness of matrices
- the ambient space of variables (e.g., \( \mathbf{R}^n \))
- linear, affine, or convex nature of constraints
- domain and codomain of functions
- basic prerequisites required before an object used in the problem is meaningful

However, you must strictly follow the principles below:

### 1. Do not add properties that are already logically derivable from the given assumptions
- If a matrix is already stated to be symmetric positive definite, do not additionally state that it is invertible.
- If a function is already stated to be twice continuously differentiable, do not additionally state that it is differentiable.
- If a conclusion already follows directly from the definitions or assumptions, do not restate it as a new assumption.

### 2. Only add conditions that are necessary and reasonable; do not over-strengthen the problem
Always prefer the weakest sufficient condition consistent with the intended exercise.
- Do not strengthen "convex" to "strictly convex" or "strongly convex" without necessity.
- Do not arbitrarily introduce Slater's condition, Lipschitz continuity, uniqueness of solution, strong duality, or similar stronger assumptions unless the problem clearly requires them.
- Do not add assumptions merely to make the problem look nicer.

### 3. If the original problem is already complete, change as little as possible
If the original statement is already self-contained, or needs only very minor clarification, make only minimal edits.

### Decision Rule
Before adding any condition, check:
1. Is this condition already explicitly stated?
2. Is this condition logically derivable from the current assumptions?
3. Is this condition necessary for the problem to be well-posed?

Add a condition only if: (1) is No, (2) is No, and (3) is Yes.

--------------------------------
[Hard Constraints]
--------------------------------

- Do NOT solve the problem.
- Do NOT add any assumptions, definitions, domains, properties, or new symbols beyond what is explicit in the input or clearly necessary for well-posedness.
- Do NOT add new tasks/questions/sub-questions.
- Do NOT change the task type (prove/show/find/derive/evaluate/minimize/etc.).
- Do NOT remove essential formulas, quantifiers, constraints, variables, domains, or conditions.
- Keep the original meaning and task unchanged; standardize wording, notation, and completeness.
- Remove non-mathematical filler and redundant narrative words, but do NOT remove mathematical content.
- Prefer concise proposition-style mathematical phrasing.
- Use formal textbook-style English.
- Use STANDARD LaTeX math notation only; no Unicode math symbols.
  Use \le, \ge, \ne, \in, \subseteq, \to, \times, \cdot, \nabla, \mathbf{R}, \mathbf{N}, \mathbf{Z}.
- Keep formulas compilable and avoid malformed delimiters.
- Keep semantic equivalence exactly: do not strengthen/weaken assumptions, claims, or quantifier scope beyond the condition completion described above.
- **ZERO REFERENCES IN OUTPUT**: The final output MUST NOT contain any equation reference numbers (e.g., (11.35)), theorem/lemma/algorithm labels (e.g., Theorem 11.8, Algorithm 11.5), or any pointer of the form "(X.Y)", "Eq. (N)", "see (N)", "by Theorem N", etc. Every such pointer must be fully resolved into its mathematical content.
- **SEMANTIC FLUENCY**: The output must read as a single, coherent, self-contained mathematical problem statement in fluent English. Inserted content must be grammatically integrated into the surrounding text, not simply appended or listed separately.

--------------------------------
[Output Format]
--------------------------------

Return STRICT JSON only, with exactly one key:

{"problem_standardized_math": "<condition-complete, self-contained, standardized mathematical exercise statement>"}

Do not output explanation, markdown, or extra keys.
Do not output anything outside the JSON.
