import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv


# Load environment
def load_env_config():
    """
    Load configuration from .env file in current directory.
    """
    # Load .env file
    load_dotenv()
    load_dotenv(".local.env", override=True)

    return {
        "api_base": os.getenv("REWARD_API_BASE"),
        "api_key": os.getenv("REWARD_API_KEY"),
        "model_name": os.getenv("REWARD_MODEL_NAME", "")
    }

# Load configuration
config = load_env_config()

MAX_CONCURRENT_REQUESTS = 16
MAX_RETRIES = 3
BASE_DELAY = 10
TIMEOUT = 150.0
_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
_llm_http_client: Optional[httpx.AsyncClient] = None

# Setup logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def _safe_preview(text: str, limit: int = 1200) -> str:
    if text is None:
        return "None"
    return text if len(text) <= limit else text[:limit] + "...(truncated)"

async def _get_llm_client() -> httpx.AsyncClient:
    global _llm_http_client
    if _llm_http_client is None or _llm_http_client.is_closed:
        _llm_http_client = httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT))
    return _llm_http_client


async def close_llm_client():
    global _llm_http_client
    if _llm_http_client is not None and not _llm_http_client.is_closed:
        await _llm_http_client.aclose()
    _llm_http_client = None


async def call_llm_api(
    user_prompt: str,
    system_prompt: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: str = "",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    response_format: Optional[Dict] = None,
):
    async with _semaphore:
        if not api_base:
            raise ValueError("api_base is required")
        if not api_key:
            raise ValueError("api_key is required")

        url = f"{api_base.rstrip('/')}/chat/completions"

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        client = await _get_llm_client()

        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )

                if resp.status_code >= 400:
                    logger.warning(
                        "LLM API non-2xx (attempt %d/%d). status=%s url=%s model=%s body=%s",
                        attempt,
                        MAX_RETRIES,
                        resp.status_code,
                        url,
                        model_name,
                        _safe_preview(resp.text),
                    )

                resp.raise_for_status()

                try:
                    data = resp.json()
                except Exception as je:
                    logger.error(
                        "LLM API invalid JSON (attempt %d/%d). status=%s url=%s model=%s raw=%s err=%r",
                        attempt,
                        MAX_RETRIES,
                        resp.status_code,
                        url,
                        model_name,
                        _safe_preview(resp.text),
                        je,
                    )
                    raise

                try:
                    content = data["choices"][0]["message"]["content"]
                except Exception as pe:
                    logger.error(
                        "LLM API unexpected schema (attempt %d/%d). url=%s model=%s data_preview=%s err=%r",
                        attempt,
                        MAX_RETRIES,
                        url,
                        model_name,
                        _safe_preview(json.dumps(data, ensure_ascii=False)),
                        pe,
                    )
                    raise

                if content is None:
                    raise ValueError("API returned None as content")

                return content

            except Exception as e:
                last_exc = e
                if attempt < MAX_RETRIES:
                    delay = BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "call_llm_api failed (attempt %d/%d). url=%s model=%s err=%r; retrying in %.2fs",
                        attempt,
                        MAX_RETRIES,
                        url,
                        model_name,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Failed after %d attempts. url=%s model=%s last_err=%r",
                        MAX_RETRIES,
                        url,
                        model_name,
                        e,
                    )

        return "No message"

async def rubrics_to_score(question: str, solution: str, rubrics: str):
    """
    Use LLM to judge if the predicted answer matches the ground truth.
    Async version.
    """
    if "###STOP" not in solution and "###TRANSFER_TO_HUMAN" not in solution:
        return 0.0
    
    user_prompt = f"""# Task Description

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
When checking **Policy Compliance**, Check whether the solution directly violates the policy — for example, if using tool A is explicitly forbidden, but the solution trace still uses tool A. However, if the steps are unrelated to the policy — for example, the policy forbids using tool A for task B, but the trace never uses tool A at all — then it is not a violation.


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
    try:
        response = await call_llm_api(
            user_prompt=user_prompt,
            system_prompt="You are a very strict evaluator who only approves of perfect answers.",
            api_base=config["api_base"],
            api_key=config["api_key"],
            model_name=config["model_name"],
            max_tokens=10240,
            temperature=0.1,
            response_format=None
        )
        result: dict[str, list[Any]] = {
            'section_scores': [],
        }
        section_scores_match = re.search(r'<section_status>(.*?)</section_status>', response, re.DOTALL)
        if section_scores_match:
            section_content = section_scores_match.group(1)
            
            sections = re.findall(r'### Section \d+.*?\*\*Name:\*\* (.*?)\n\*\*Completed Items:\*\* (\d+)/(\d+)', 
                                section_content, re.DOTALL)
            
            for section_name, score, total in sections:
                result['section_scores'].append({
                    'name': section_name.strip(),
                    'score': int(score),
                    'total': int(total)
                })
        
        score = 0
        total_sec_count = 0
        for section in result['section_scores']:
            if section["name"].strip().startswith("Policy") or section["name"].strip().startswith("Concise"):
                if section['total'] == 0 or section['score'] / section['total'] != 1:
                    return 0.0
                continue

            if section['total'] != 0:
                if section["name"].strip().startswith("Task"):
                    score += section['score'] / section['total']
    
                total_sec_count += 1

        return score if total_sec_count > 0 else 0.3
    
    except Exception as e:
        logger.error(f"Error in rubrics_to_score: {e}")
        return 0.0

def _clean_solution_for_judge(solution_str):
    """Clean solution_str for reward judging.
    
    Keep only: user messages, agent's <question> content, <tool_call> content,
    tool responses (wrapped in <tool_response> inside user turns), <answer> content,
    and ###STOP / ###TRANSFER_TO_HUMAN markers.
    Strip out: system messages, <think> blocks, and other assistant reasoning text.
    
    Note: there is no separate 'tool' role. Tool responses appear as user turns
    containing <tool_response>...</tool_response> tags.
    """
    # Remove system messages
    pattern = r'<\|im_start\|>system\n.*?<\|im_end\|>\n?'
    cleaned = re.sub(pattern, '', solution_str, flags=re.DOTALL)

    # Parse all turns (only user and assistant roles exist)
    turns = re.findall(r'<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>', cleaned, flags=re.DOTALL)

    parts = []
    turn_number = 0
    for role, content in turns:
        if role == 'user':
            # Distinguish real user messages from tool responses
            tool_responses = re.findall(r'<tool_response>(.*?)</tool_response>', content, flags=re.DOTALL)
            if tool_responses:
                for tr in tool_responses:
                    parts.append(f"[Tool Response]:\n{tr.strip()}")
            else:
                turn_number += 1
                parts.append(f"[Round {turn_number} - User]:\n{content.strip()}")
        elif role == 'assistant':
            # Extract <question>, <tool_call>, and <answer> content
            questions = re.findall(r'<question>(.*?)</question>', content, flags=re.DOTALL)
            tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', content, flags=re.DOTALL)
            answers = re.findall(r'<answer>(.*?)</answer>', content, flags=re.DOTALL)

            for q in questions:
                parts.append(f"[Round {turn_number} - Agent Question]:\n{q.strip()}")
            for tc in tool_calls:
                parts.append(f"[Tool Call - Agent]:\n{tc.strip()}")
            for a in answers:
                parts.append(f"[Agent Answer]:\n{a.strip()}")

            # Preserve termination markers
            if "###TRANSFER_TO_HUMAN" in content:
                parts.append("[Agent Escalated to Human]: ###TRANSFER_TO_HUMAN")

    return '\n\n'.join(parts)


async def compute_score_virtual_tool(data_source, solution_str, ground_truth, extra_info, keep_genrm_critics=False, **kwargs):
    """
    Compute score for RL training. Fully async version.
    """
    question = extra_info["question"]
    rubrics = extra_info["rubrics"]

    score = await rubrics_to_score(
        question=question,
        solution=solution_str,
        rubrics=rubrics
    )
    return score


async def compute_score_virtual_tool_v3(data_source, solution_str, ground_truth, extra_info, keep_genrm_critics=False, **kwargs):
    """
    Compute score for RL training. Fully async version.
    """
    if ("###STOP" not in solution_str and "###TRANSFER_TO_HUMAN" not in solution_str) or "<tool_response>" not in solution_str:
        return 0.0

    question = extra_info["question"]
    rubrics = extra_info["rubrics"]

    # Clean solution_str: keep only question, tool call, tool response and final answer
    cleaned_solution = _clean_solution_for_judge(f"<|im_start|>assistant\n{solution_str}")

    judge_prompt = f"""You are a **Task Completion Judge**, responsible for strictly evaluating whether an agent has successfully completed a task.

You will receive the following inputs:

1. **The conversation history between the user and the intelligent agent**  
2. **The full history of tool calls made by the agent**  
   (including each tool’s inputs and outputs)  
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
You must list **all task‑relevant database fields** and provide:

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
    response = await call_llm_api(
        user_prompt=judge_prompt,
        system_prompt="You are a very strict evaluator who only approves of perfect answers.",
        api_base=config["api_base"],
        api_key=config["api_key"],
        model_name=config["model_name"],
        max_tokens=10240,
        temperature=0.1,
        response_format=None
    )
    judge_score_match = re.search(r'<final_judgment>(.*?)</final_judgment>', response, re.DOTALL)
    if judge_score_match:
        judge_score = judge_score_match.group(1)
        if "task completed" in judge_score.strip().lower():
            return 1.0
        else:
            return 0.0
    else:
        return 0.0


async def compute_score_virtual_tool_v3_2(data_source, solution_str, ground_truth, extra_info, keep_genrm_critics=False, **kwargs):
    """
    Compute score for RL training. Fully async version.
    """
    if ("###STOP" not in solution_str and "###TRANSFER_TO_HUMAN" not in solution_str) or "<tool_response>" not in solution_str:
        return 0.0

    import json5
    question = extra_info["question"]
    policy_str = extra_info["policy"]

    # Clean solution_str: keep only question, tool call, tool response and final answer
    cleaned_solution = _clean_solution_for_judge(solution_str)

    reference_success_or_fail_path = json5.loads(extra_info["tools_kwargs"]["mock_tool"]["create_kwargs"]["tool_return_expected"])
    success_path = reference_success_or_fail_path["normal_path"]
    fail_path = reference_success_or_fail_path["hack_path"]

    judge_prompt = f"""You are a **Task Completion Judge**, responsible for strictly evaluating whether an agent has successfully completed a task.

You will be given:

1. The **conversation history** between the user and the agent  
2. The agent’s full **tool-call history** (including inputs and outputs for each call)  
3. The task **policy**, specifying when the task **must** be completed and when it **must not** be completed

You may also receive example “possibly correct” and “possibly incorrect” completion path for reference. Note that “possibly correct” traces are **not guaranteed** to be correct.

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

- reservation.status: initial = "active"; after tool calls = "active" (not cancelled); complies with policy—based on context and policy, this state should not be changed via tools

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
{policy_str}

The question is:
{question}

The tool call and interaction history between the user and the agent is:
{cleaned_solution}

The possibly correct path is:
{success_path}

The possibly incorrect path is:
{fail_path}
"""
    response = await call_llm_api(
        user_prompt=judge_prompt,
        system_prompt="You are a very strict evaluator who only approves of perfect answers.",
        api_base=config["api_base"],
        api_key=config["api_key"],
        model_name=config["model_name"],
        max_tokens=10240,
        temperature=0.1,
        response_format=None
    )
    judge_score_match = re.search(r'<final_judgment>(.*?)</final_judgment>', response, re.DOTALL)
    if judge_score_match:
        judge_score = judge_score_match.group(1)
        if "success" in judge_score.strip().lower():
            return 1.0
        else:
            return 0.0
    else:
        return 0.0


async def judge_answer_correctness(question: str, predicted_answer: str, ground_truth: str) -> bool:
    """
    Use LLM to judge if the predicted answer matches the ground truth.
    Async version to enable concurrent judging.
    """
    system_prompt = """You will be given a question, a solution and a ground truth answer.
Your task is to determine if the answer that the solution gives is equal to the ground truth answer, ignoring minor differences in formatting, spacing, or notation.
You must respond in JSON format with a 'equivalent' field that is either true or false."""
    
    user_prompt = f"""Question: {question}

Solution: {predicted_answer}
    
Ground truth answer: {ground_truth}
    
Are the answer that the solution gives and the ground truth answer semantically equivalent? Respond in JSON format with only the 'equivalent' field.

Please be strict and accurate!!!

Example response: {{"equivalent": true}}
"""
# Sometimes, predicted_answer may contain some irrelevant content, please ignore it, as long as predicted_answer contains the final answer, it is considered correct.
# Example: predicted_answer: {{The two sets are different because the sum of the remainders cannot equal the sum of the integers under the given constraints.}}. ground_truth: {{The two sets are different.}}. predicted_answer should be considered correct.

    try:
        response = await call_llm_api(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            api_base=config["api_base"],
            api_key=config["api_key"],
            model_name=config["model_name"],
            max_tokens=50,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        # Check if response contains "true"
        return "true" in response.strip().lower()
        
    except Exception as e:
        # If there's any error in calling the API or parsing JSON, 
        # fall back to strict string comparison
        logger.error(f"Error in judge_answer_correctness: {e}")
        return predicted_answer.strip() == str(ground_truth).strip()

async def compute_score_math(data_source, solution_str, ground_truth, extra_info, keep_genrm_critics=False, **kwargs):
    # Extract question from extra info
    question = extra_info["question"]
    
    # Extract answer from solution string using regex
    # Looks for content within <answer></answer> tags
    answer_matches = re.findall(r"<answer>(.+?)</answer>", solution_str, re.DOTALL)
    if answer_matches:
        # Use the last matched answer if multiple are found
        answer_str = answer_matches[-1]
    else:
        # If no tags found, use entire solution string
        return 0.0
    
    # Async judge whether answer is correct
    is_correct = await judge_answer_correctness(question, answer_str, ground_truth)
    
    # Assign reward score
    if is_correct:
        if "<tool_call>" in solution_str:
            reward_score = 1.0
        else:
            reward_score = 0.9
    else:
        reward_score = 0.0  # Format reward for incorrect answers
    
    return reward_score

