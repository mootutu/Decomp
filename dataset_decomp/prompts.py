"""Prompts for dataset decomposition."""

DECOMPOSITION_SYSTEM = """You are a rigorous math dataset builder.
Return only valid JSON. Do not use Markdown fences.
Every generated subproblem must be easier than the parent problem and must have a verifiable numeric or short symbolic answer."""


ORIGINAL_SOLUTION_PROMPT = """Solve this AIME problem step by step.

Problem:
{problem}

Known final answer: {answer}

Return this JSON object:
{{
  "solution": "concise step-by-step solution that reaches the known answer",
  "final_answer": "{answer}"
}}"""


DECOMPOSE_PROMPT = """We are reproducing dataset decomposition for complex math problems.

Parent problem:
{problem}

Parent final answer:
{answer}

Parent solution:
{solution}

Decompose the solution into prerequisite concepts and simpler subproblems.
Each subproblem should teach or test one step needed by the parent solution.
Prefer 3 to 6 subproblems unless the parent is already simple.

Return this JSON object exactly:
{{
  "summary": "one sentence describing the parent solution strategy",
  "concept_tags": ["short concept tag", "..."],
  "steps": [
    {{
      "step_id": "s1",
      "description": "what this step proves or computes",
      "concept_tags": ["tag"],
      "subproblem": "self-contained easier math problem",
      "expected_answer": "numeric or short symbolic answer",
      "solution": "short solution to the subproblem",
      "depends_on": []
    }}
  ],
  "is_atomic": false
}}

Set "is_atomic" to true and use an empty steps list if no meaningful simpler subproblems are needed."""


VERIFY_PROMPT = """Verify this generated math subproblem.

Subproblem:
{problem}

Proposed answer:
{answer}

Proposed solution:
{solution}

Return this JSON object:
{{
  "valid": true,
  "corrected_answer": "answer if correction is needed, otherwise same as proposed",
  "corrected_solution": "solution if correction is needed, otherwise same as proposed",
  "reason": "brief verification rationale"
}}"""

