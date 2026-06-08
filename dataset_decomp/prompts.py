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


SOLVE_SUBPROBLEM_PROMPT = """Solve this generated math subproblem independently.

Subproblem:
{problem}

Return this JSON object:
{{
  "solution": "concise step-by-step solution",
  "final_answer": "numeric or short symbolic answer"
}}"""


REGENERATE_SUBPROBLEM_PROMPT = """Regenerate one failed generated math subproblem.

Parent problem:
{parent_problem}

Parent final answer:
{parent_answer}

Parent solution:
{parent_solution}

Target decomposition step:
{step_description}

Target concept tags:
{concept_tags}

Previous invalid subproblem:
{invalid_problem}

Previous proposed answer:
{invalid_answer}

Previous proposed solution:
{invalid_solution}

Verification failure reason:
{failure_reason}

Create a replacement subproblem for the same target step. It must be self-contained,
easier than the parent problem, and have a unique numeric or short symbolic answer.

Return this JSON object exactly:
{{
  "step_id": "{step_id}",
  "description": "what this step proves or computes",
  "concept_tags": ["tag"],
  "subproblem": "self-contained easier math problem",
  "expected_answer": "numeric or short symbolic answer",
  "solution": "short solution to the subproblem",
  "depends_on": []
}}"""
