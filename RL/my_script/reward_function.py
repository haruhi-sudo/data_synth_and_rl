import logging
import os
import re
from typing import Any

from dotenv import load_dotenv

from my_script.prompts.judge import (
    ANSWER_JUDGE_SYSTEM_PROMPT,
    ANSWER_JUDGE_USER_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_V3_2_PROMPT,
    JUDGE_V3_PROMPT,
    RUBRICS_JUDGE_PROMPT,
)
from my_script.utils.llm_client import call_llm_async, close_async_client
from my_script.utils.message_parser import extract_solution_summary


# Load environment
load_dotenv()
load_dotenv(".local.env", override=True)

config = {
    "api_base": os.getenv("REWARD_API_BASE"),
    "api_key": os.getenv("REWARD_API_KEY"),
    "model_name": os.getenv("REWARD_MODEL_NAME", ""),
}

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


async def rubrics_to_score(question: str, solution: str, rubrics: str):
    """Use LLM to evaluate solution against rubrics. Returns a float score."""
    if "###STOP" not in solution and "###TRANSFER_TO_HUMAN" not in solution:
        return 0.0

    user_prompt = RUBRICS_JUDGE_PROMPT.format(
        question=question,
        rubrics=rubrics,
        solution=solution,
    )
    try:
        response = await call_llm_async(
            user_prompt=user_prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            api_base=config["api_base"],
            api_key=config["api_key"],
            model_name=config["model_name"],
            max_tokens=10240,
            temperature=0.1,
        )
        result: dict[str, list[Any]] = {'section_scores': []}
        section_scores_match = re.search(r'<section_status>(.*?)</section_status>', response, re.DOTALL)
        if section_scores_match:
            section_content = section_scores_match.group(1)
            sections = re.findall(
                r'### Section \d+.*?\*\*Name:\*\* (.*?)\n\*\*Completed Items:\*\* (\d+)/(\d+)',
                section_content, re.DOTALL,
            )
            for section_name, score, total in sections:
                result['section_scores'].append({
                    'name': section_name.strip(),
                    'score': int(score),
                    'total': int(total),
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


async def compute_score_virtual_tool(data_source, solution_str, ground_truth, extra_info, keep_genrm_critics=False, **kwargs):
    """Compute score for RL training (v1 rubrics-based)."""
    question = extra_info["question"]
    rubrics = extra_info["rubrics"]
    return await rubrics_to_score(question=question, solution=solution_str, rubrics=rubrics)


async def compute_score_virtual_tool_v3(data_source, solution_str, ground_truth, extra_info, keep_genrm_critics=False, **kwargs):
    """Compute score for RL training (v3 task completion judge)."""
    if ("###STOP" not in solution_str and "###TRANSFER_TO_HUMAN" not in solution_str) or "<tool_response>" not in solution_str:
        return 0.0
    # breakpoint()
    question = extra_info["question"]
    rubrics = extra_info["rubrics"]
    cleaned_solution = extract_solution_summary(f"<|im_start|>assistant\n{solution_str}")

    user_prompt = JUDGE_V3_PROMPT.format(
        question=question,
        cleaned_solution=cleaned_solution,
        rubrics=rubrics,
    )
    response = await call_llm_async(
        user_prompt=user_prompt,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        api_base=config["api_base"],
        api_key=config["api_key"],
        model_name=config["model_name"],
        max_tokens=10240,
        temperature=0.1,
    )
    judge_score_match = re.search(r'<final_judgment>(.*?)</final_judgment>', response, re.DOTALL)
    if judge_score_match:
        judge_score = judge_score_match.group(1)
        return 1.0 if "task completed" in judge_score.strip().lower() else 0.0
    return 0.0


async def compute_score_virtual_tool_v3_2(data_source, solution_str, ground_truth, extra_info, keep_genrm_critics=False, **kwargs):
    """Compute score for RL training (v3.2 policy-aware judge)."""
    if ("###STOP" not in solution_str and "###TRANSFER_TO_HUMAN" not in solution_str) or "<tool_response>" not in solution_str:
        return 0.0

    import json5
    question = extra_info["question"]
    policy_str = extra_info["policy"]
    cleaned_solution = extract_solution_summary(solution_str)

    reference_success_or_fail_path = json5.loads(extra_info["tools_kwargs"]["mock_tool"]["create_kwargs"]["tool_return_expected"])
    success_path = reference_success_or_fail_path["normal_path"]
    fail_path = reference_success_or_fail_path["hack_path"]

    user_prompt = JUDGE_V3_2_PROMPT.format(
        policy=policy_str,
        question=question,
        cleaned_solution=cleaned_solution,
        success_path=success_path,
        fail_path=fail_path,
    )
    response = await call_llm_async(
        user_prompt=user_prompt,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        api_base=config["api_base"],
        api_key=config["api_key"],
        model_name=config["model_name"],
        max_tokens=10240,
        temperature=0.1,
    )
    judge_score_match = re.search(r'<final_judgment>(.*?)</final_judgment>', response, re.DOTALL)
    if judge_score_match:
        judge_score = judge_score_match.group(1)
        return 1.0 if "success" in judge_score.strip().lower() else 0.0
    return 0.0


async def judge_answer_correctness(question: str, predicted_answer: str, ground_truth: str) -> bool:
    """Use LLM to judge if the predicted answer matches the ground truth."""
    try:
        response = await call_llm_async(
            user_prompt=ANSWER_JUDGE_USER_PROMPT.format(
                question=question,
                predicted_answer=predicted_answer,
                ground_truth=ground_truth,
            ),
            system_prompt=ANSWER_JUDGE_SYSTEM_PROMPT,
            api_base=config["api_base"],
            api_key=config["api_key"],
            model_name=config["model_name"],
            max_tokens=50,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return "true" in response.strip().lower()
    except Exception as e:
        logger.error(f"Error in judge_answer_correctness: {e}")
        return predicted_answer.strip() == str(ground_truth).strip()


async def compute_score_math(data_source, solution_str, ground_truth, extra_info, keep_genrm_critics=False, **kwargs):
    """Compute score for math RL training."""
    question = extra_info["question"]

    answer_matches = re.findall(r"<answer>(.+?)</answer>", solution_str, re.DOTALL)
    if answer_matches:
        answer_str = answer_matches[-1]
    else:
        return 0.0

    is_correct = await judge_answer_correctness(question, answer_str, ground_truth)

    if is_correct:
        return 1.0 if "<tool_call>" in solution_str else 0.9
    return 0.0
