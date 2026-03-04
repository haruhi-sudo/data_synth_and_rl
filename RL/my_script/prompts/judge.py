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
# V3: Task completion judge (used by compute_score_virtual_tool_v3)
# ---------------------------------------------------------------------------

JUDGE_V3_PROMPT = """You are a **Task Completion Judge**, responsible for strictly evaluating whether an agent has successfully completed a task.

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
# V3.2: Policy-aware judge (used by compute_score_virtual_tool_v3_2)
# ---------------------------------------------------------------------------

JUDGE_V3_2_PROMPT = """You are a **Task Completion Judge**, responsible for strictly evaluating whether an agent has successfully completed a task.

You will be given:

1. The **conversation history** between the user and the agent
2. The agent's full **tool-call history** (including inputs and outputs for each call)
3. The task **policy**, specifying when the task **must** be completed and when it **must not** be completed

You may also receive example "possibly correct" and "possibly incorrect" completion path for reference. Note that "possibly correct" traces are **not guaranteed** to be correct.

## Output Requirements

### 1. Reasoning
Provide step-by-step reasoning, including:

- The relevant database state *before* any tool calls
- How each tool call changed (or failed to change) the database state
- Whether each tool call complies with the policy
- If it complies, whether the database was updated to the **correct** values

Reference correct or incorrect path may be used, but they are not authoritative.

Your reasoning must be detailed, explicit, and include all key inferences.

### 2. State Evolution
List **all database fields relevant to the task**, and for each provide:

- The initial value (inferred from tools and/or evaluation criteria)
- The effect of each tool call on the value
- The final value
- Whether the final value satisfies the policy

Example format:

<state_evolution>

- reservation.status: initial = "active"; after tool calls = "active" (not cancelled); complies with policy\u2014based on context and policy, this state should not be changed via tools

- reservation.flight_number: initial = "SSS001"; after tool calls = "AAA001"; complies with policy

</state_evolution>

### 3. Final Judgment
Output exactly one sentence:

<final_judgment>

{{"judgment": "success" or "failed"}}

</final_judgment>

Where **success** means:

1) The task **should** be completed, and the database has been updated to the correct values; or
2) The task **should not** be completed, and the database remains unchanged; or
3) Some parts **should** be completed and others **should not**: the required fields were updated correctly, and fields that should not change remain unchanged.

--------
The policy is:
{policy}

The question is:
{question}

The tool call and interaction history between the user and the agent is:
{cleaned_solution}

The possibly correct path is:
{success_path}

The possibly incorrect path is:
{fail_path}
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
