"""Prompt templates for reward judging (all versions)."""

JUDGE_SYSTEM_PROMPT = "You are a very strict evaluator who only approves of perfect answers."

# ---------------------------------------------------------------------------
# V1: Rubrics-based judge (used by compute_score_virtual_tool)
# ---------------------------------------------------------------------------

RUBRICS_JUDGE_PROMPT = """# Task Description

{question}

---

# Assessment Rubrics

{rubrics}

---

# Solution to Evaluate

{solution}

---

# Your Task

Evaluate the above solution based on the provided rubrics. For each criterion, determine whether the solution is:

- Completed
- Not Completed

### Special Reminder (Important)
When checking **Policy Compliance**, Check whether the solution directly violates the policy \u2014 for example, if using tool A is explicitly forbidden, but the solution trace still uses tool A. However, if the steps are unrelated to the policy \u2014 for example, the policy forbids using tool A for task B, but the trace never uses tool A at all \u2014 then it is not a violation.


This must be explicitly reflected in the evaluation.

---

# Output Format

<evaluation>
## Section 1 Policy Compliance Criteria
### [Criterion 1 Name]
**Status: Completed / Not Completed**
[Explain why this criterion is completed or not completed. f the agent's behavior is not related to the policy, it is not considered as a violation.]

## Section 2 Task Sub-goals and Required User Interaction
### [Criterion 1 Name]
**Status: Completed / Not Completed**
[Explain why this criterion is completed or not completed.]

### [Criterion 2 Name]
**Status: Completed / Not Completed**
[Explain why this criterion is completed or not completed.]

[Continue for all criteria...]

</evaluation>

<section_status>
### Section 1
**Name:** Policy Compliance Criteria
**Completed Items:** X/Y

### Section 2
**Name:** Task Sub-goals and Required User Interaction
**Completed Items:** X/Y

</section_status>

---

Please evaluate all the criterion, and be strict! Begin your evaluation now and follow the output format exactly.
Be strict, and be strict!
"""

# ---------------------------------------------------------------------------
# V3: Task completion judge (used by compute_score_virtual_tool_completion)
# ---------------------------------------------------------------------------

RUBRICS_JUDGE_01_PROMPT = """You are a **Task Completion Judge**, responsible for strictly evaluating whether an agent has successfully completed a task.

You will receive the following inputs:

1. **The conversation history between the user and the intelligent agent**
2. **The full history of tool calls made by the agent**
   (including each tool's inputs and outputs)
3. **A task evaluation rubric**, which specifies the criteria for determining completion.
   - The rubric defines which database fields must be updated to which specific values
   - Or which fields must remain unchanged
   - These rules determine whether the task should be considered successful or failed

Your job is to determine whether the task has been successfully completed, based solely on the objective criteria described in the rubric.

------------------------------------------------------------
### Output Requirements

You must output the following three sections:

#### 1. Reasoning
In this section, you must provide step-by-step reasoning that includes:

- The relevant database state *before* any tool calls
- How each tool call may modify or fail to modify the database state
- Whether the tool outputs logically imply a state change
- The exact state transitions and their consistency with the rubric
- Identification of any mismatch between actual outcomes and required rubric conditions

The reasoning must be thorough, explicit, and include every critical inference.

#### 2. State Evolution
You must list **all task\u2011relevant database fields** and provide:

- Their initial values (as inferred from the tools or the rubric)
- The effect of each tool call on those values
- Their final values
- Whether each final value satisfies the rubric requirement

Example format:
```
<state_evolution>
- reservation.status: initial = "active"; after tool calls = "active" (not cancelled); matches rubric
- reservation.flight_number: initial = "SSS001"; after tool calls = "AAA001"; matches rubric
</state_evolution>
```

#### 3. Final Judgment
You must output a single sentence:

<final_judgment>
{{"judgment": "Task completed" or "Task not completed"}}
</final_judgment>

--------
The question is:
{question}

The tool call and interaction history between the user and the agent is:
{cleaned_solution}

The rubric is:
{rubrics}
"""

# ---------------------------------------------------------------------------
# Answer correctness judge (used by compute_score_math)
# ---------------------------------------------------------------------------

ANSWER_JUDGE_SYSTEM_PROMPT = """You will be given a question, a solution and a ground truth answer.
Your task is to determine if the answer that the solution gives is equal to the ground truth answer, ignoring minor differences in formatting, spacing, or notation.
You must respond in JSON format with a 'equivalent' field that is either true or false."""

ANSWER_JUDGE_USER_PROMPT = """Question: {question}

Solution: {predicted_answer}

Ground truth answer: {ground_truth}

Are the answer that the solution gives and the ground truth answer semantically equivalent? Respond in JSON format with only the 'equivalent' field.

Please be strict and accurate!!!

Example response: {{"equivalent": true}}
"""
