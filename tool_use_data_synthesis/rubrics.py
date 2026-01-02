import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from functions import call_llm_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a professional AI system evaluation expert specializing in assessing the quality of tool-use task completion.

Your responsibilities:
1. Compare different solution trajectories and analyze their strengths and weaknesses
2. Based on the evaluation, develop detailed and generalizable assessment rubrics for the given task
3. Judge each solution based on the rubrics and provide clear conclusions with the best solution

**Important**: Please strictly follow the specified XML tag format for output to facilitate subsequent parsing and saving."""


def create_user_prompt(trajectories_summary: str, task_description: str, restrict_policy: str, task_background: str, high_level_workflow: str) -> str:
    """Create user prompt for comparison."""
    
    return f"""# Task Description

{task_description}

---

# Task Background

{task_background}

---

# High-Level Workflow (Planned Sub-goals)

{high_level_workflow}

This represents the intended solution path designed when the task was created. Solutions should generally align with this workflow, though minor deviations are acceptable if well-justified.

---

# The Policy the Agent Should Follow

{restrict_policy}

---

# Agent Solution Trajectories to Compare

{trajectories_summary}

---

# Your Evaluation Task

Please complete the evaluation following these steps and **strictly use the specified XML tags** to wrap each section:

## Step 1: Workflow Alignment Check

Place your workflow alignment analysis within <alignment_check></alignment_check> tags.

Evaluate the overall alignment between the provided solutions and the planned high-level workflow:
- Do the solutions collectively cover the intended sub-goals from the workflow (using different approaches is allowed)?
- Are the solutions addressing the same core problem that the workflow was designed to solve?
- Is the collective execution pattern reasonably aligned with the workflow design, or does it indicate a fundamental mismatch?

**Decision Rule**: If the solutions collectively cover most of the planned workflow sub-goals (even if each individual solution only covers part of it), the task should be kept. Only discard if there is a fundamental mismatch where the workflow design doesn't align with what the task actually requires.

Conclude with one of:
- ✅ **KEEP**: Solutions collectively align with the workflow (partial coverage by individual solutions is acceptable)
- ❌ **DISCARD**: Fundamental misalignment between workflow and executions indicates flawed task design - discard this entire data point

---

## Step 2: Comparative Analysis of Valid Solutions

Place your comparative analysis within <reasoning></reasoning> tags.

**Only analyze solutions marked as KEEP from Step 1.**

For each valid solution, evaluate:
- Strengths and weaknesses  
- Whether it violates the policy directly  
- Whether it makes assumptions instead of asking the user  
- Whether required sub-goals (from high-level workflow) are completed  
- Whether it interacts sufficiently with the user to gather missing information  
- How well it follows the planned workflow path
- Overall logic, quality, and practicality

Use markdown formatting for clarity.

---

## Step 3: Develop Assessment Rubrics  
(Only define criteria—no scoring in this section.)

Place the assessment rubrics within <rubrics></rubrics> tags.

Organize your rubrics into the following four sections:

---

### 1. Policy Compliance Criteria

Based on `# The Policy the Agent Should Follow` and observed execution patterns, define **specific, task-relevant** compliance criteria.

For each specific policy, list concrete behavioral requirements the agent must follow:
  - What the policy requires
  - How to identify violations: Specific behaviors that violate this requirement
...

If an agent violates the rules, the entire task is considered a failure. Please list as many reasonable policies as possible to constrain agent behavior.

---

### 2. Task Sub-goals Completion

List all sub-goals required to complete the task successfully (reference the high-level workflow and actual execution).
For each sub-goal, clearly define:
- What the sub-goal requires (as planned in the workflow, also from the execution)
- How to determine whether the sub-goal is fully completed, or not completed

(No scoring—just criteria.)

---

### 3. Required User Interaction

List:
- All pieces of information the agent should obtain by asking the user  
  (e.g., missing parameters, ambiguous instructions, task clarifications)

For each item, define:
- What the agent must ask  
- How to determine whether the interaction was sufficient  
- What behaviors count as prematurely acting or making unsupported assumptions

---

## Step 4: Provide Final Conclusion

Place the final conclusion within <final></final> tags.

Provide a **qualitative** comparison of all solutions based on your rubrics:
- Whether each solution follows policy  
- Whether sub-goals are completed  
- Whether necessary user interaction occurred  

Then identify which solution is best among the valid ones and explain why.

Finally, output only the filename of the best solution within:
<best_solution>
solution_x.json
</best_solution>

---

**Output Format Requirements:**

<alignment_check>
[Workflow alignment analysis for each solution, with KEEP/DISCARD decisions]
</alignment_check>

<reasoning>
[Comparative analysis of KEEP solutions only, using markdown]
</reasoning>

<rubrics>
[Rubrics here — criteria only, organized in 3 sections: 1. Policy, 2. Task Sub-goals, 3. Required User Interaction]
</rubrics>

<final>
[Qualitative comparison of valid solutions and best-solution selection]
</final>

<best_solution>
[Best solution filename]
</best_solution>

Please begin your evaluation now."""


def create_comparison_prompt(trajectories_summary: str, task_description: str, restrict_policy: str, task_background: str, high_level_workflow: str) -> Tuple[str, str]:
    """
    Create comparison prompts.
    
    Args:
        trajectories_summary: Formatted summary of all trajectories
        task_description: Task description
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    user_prompt = create_user_prompt(trajectories_summary, task_description, restrict_policy, task_background, high_level_workflow)
    return SYSTEM_PROMPT, user_prompt

def extract_task_description(trajectory: List[Dict]) -> str:
    """Extract task description from trajectory"""
    return trajectory[1]["content"]


def count_tool_calls(trajectory: List[Dict]) -> int:
    """Count the number of tool calls in trajectory"""
    count = 0
    for message in trajectory:
        if message.get("role") == "assistant":
            content = message.get("content", "")
            count += content.count("<tool_call>")
    return count


def extract_tools_used(trajectory: List[Dict]) -> List[str]:
    """Extract list of tools used in trajectory"""
    tools = []
    for message in trajectory:
        if message.get("role") == "assistant":
            content = message.get("content", "")
            if "<tool_call>" in content:
                try:
                    start = content.find("<tool_call>") + len("<tool_call>")
                    end = content.find("</tool_call>")
                    tool_json = json.loads(content[start:end])
                    tools.append(tool_json.get("name", "unknown"))
                except:
                    pass
    return tools


def format_trajectory_for_comparison(file_name: str, trajectory: List[Dict]) -> str:
    """Format a single trajectory for comparison"""
    tool_count = count_tool_calls(trajectory)
    tools_used = extract_tools_used(trajectory)
    
    # Simplify trajectory, keep only key information
    simplified = f"### File: {file_name}\n\n"
    simplified += f"**Tool Call Count**: {tool_count}\n"
    simplified += f"**Tools Used**: {', '.join(tools_used)}\n\n"
    simplified += "**Trajectory Summary**:\n"
    
    for i, message in enumerate(trajectory):
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role == "system":
            continue  # Skip system message (too long)
        elif role == "user":
            if i == 1:
                simplified += f"\n[User Request]\n{content[:800]}...\n"
            elif "tool_response" in content or role == "tool":
                simplified += f"\n[Tool Response]\n{content[:500]}...\n"
        elif role == "assistant":
            # Extract reasoning and tool calls
            if "<tool_call>" in content:
                reasoning = content.split("<tool_call>")[0].strip()
                tool_part = content[content.find("<tool_call>"):content.find("</tool_call>")+12]
                simplified += f"\n[Assistant-Step{i//2}]\nReasoning: {reasoning[:800]}...\nTool Call: {tool_part}\n"
            else:
                simplified += f"\n[Assistant-Final Answer]\n{content[:1000]}...\n"

    return simplified

def load_solution_files(folder_path: str, top_k: int = 3) -> Dict[str, List[Dict]]:
    """
    Load all JSON files starting with 'solution' from the specified folder
    
    Args:
        folder_path: Path to the folder containing solution files
    
    Returns:
        Dictionary with filenames as keys and JSON content as values
    """
    solution_files = {}
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Find all JSON files starting with 'solution'
    json_files = list(folder.glob("solution*.json"))
    
    for file_path in json_files[:top_k]: # Only load top_k files
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                solution_files[file_path.name] = content
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in {file_path.name}: {e}")
        except Exception as e:
            logger.error(f"Error reading file {file_path.name}: {e}")
    
    return solution_files

def parse_llm_response(response_content: str) -> Dict[str, str]:
    """
    Parse LLM response and extract content from XML tags
    
    Args:
        response_content: Complete LLM response
    
    Returns:
        Dictionary containing reasoning, rubrics, final, and best_solution keys
    """
    result = {
        "reasoning": "",
        "alignment_check": "",
        "rubrics": "",
        "final": "",
        "best_solution": ""
    }
    # Extract alignment_check
    alignment_check_match = re.search(r'<alignment_check>(.*?)</alignment_check>', response_content, re.DOTALL)
    if alignment_check_match:
        result["alignment_check"] = alignment_check_match.group(1).strip()
    else:
        logger.warning("Alignment check section not found in response")
    
    if result["alignment_check"] == "" or "discard" in result["alignment_check"].lower():
        return result

    # Extract reasoning
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response_content, re.DOTALL)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()
    else:
        logger.warning("Reasoning section not found in response")
    
    # Extract rubrics
    rubrics_match = re.search(r'<rubrics>(.*?)</rubrics>', response_content, re.DOTALL)
    if rubrics_match:
        result["rubrics"] = rubrics_match.group(1).strip()
    else:
        logger.warning("Rubrics section not found in response")
    
    # Extract final
    final_match = re.search(r'<final>(.*?)</final>', response_content, re.DOTALL)
    if final_match:
        result["final"] = final_match.group(1).strip()
    else:
        logger.warning("Final section not found in response")
    
    # Extract best_solution
    best_solution_match = re.search(r'<best_solution>(.*?)</best_solution>', response_content, re.DOTALL)
    if best_solution_match:
        result["best_solution"] = best_solution_match.group(1).strip()
    else:
        logger.warning("Best solution section not found in response")
    
    return result


def extract_best_solution_filename(best_solution_text: str) -> Optional[str]:
    """
    Extract filename from best_solution text
    
    Args:
        best_solution_text: Text from best_solution tag
    
    Returns:
        Extracted filename, or None if not found
    """
    if not best_solution_text:
        return None
    
    # Try to match solution*.json format
    match = re.search(r'solution[_\-]?\w*\.json', best_solution_text, re.IGNORECASE)
    if match:
        return match.group(0)
    
    # If no .json suffix, try to match solution followed by number or identifier
    match = re.search(r'solution[_\-]?\w+', best_solution_text, re.IGNORECASE)
    if match:
        filename = match.group(0)
        # Add .json suffix if missing
        if not filename.endswith('.json'):
            filename += '.json'
        return filename
    
    # If nothing matched, return cleaned text
    cleaned = best_solution_text.strip().replace('\n', ' ').replace('\r', ' ')
    if cleaned:
        return cleaned
    
    return None

def compare_trajectories(
    folder_path: str,
    solution_top_k: int,
    api_base: str,
    api_key: str,
    model_name: str,
) -> Dict:    
    # 1. Load all solution files
    solution_files = load_solution_files(folder_path, solution_top_k)
    if len(solution_files) < 2:
        logger.warning("Not enough solution files found")
        return None

    more_info_path = os.path.join(folder_path, "more_info.json")
    if not os.path.exists(more_info_path):
        return None
    else:
        with open(more_info_path, 'r', encoding='utf-8') as f:
            more_info = json.load(f)
    
    restrict_policy = more_info.get("restrict", "")
    task_background = more_info.get("task_background", "")
    high_level_workflow = more_info.get("initial_workflow", "")
    
    # 2. Extract task description (from first file)
    first_trajectory = list(solution_files.values())[0]
    task_description = extract_task_description(first_trajectory)
    
    # 3. Format all trajectories for comparison
    trajectories_summary = ""
    for file_name, trajectory in solution_files.items():
        trajectories_summary += format_trajectory_for_comparison(file_name, trajectory)
        trajectories_summary += "\n" + "="*80 + "\n\n"
    
    # 4. Create comparison prompt
    system_prompt, user_prompt = create_comparison_prompt(trajectories_summary, task_description, restrict_policy, task_background, high_level_workflow)
    
    # 5. Call LLM for comparison
    messages = call_llm_api(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        api_base=api_base,
        api_key=api_key,
        model_name=model_name,
        max_tokens=10240,
        temperature=0.3
    )
    
    # 6. Parse response
    response_content = messages[-1]["content"]
    parsed_response = parse_llm_response(response_content)
    
    # Verify all sections were extracted
    missing_parts = [k for k, v in parsed_response.items() if not v]
    if missing_parts:
        return None
    
    # Extract best solution filename
    best_solution_filename = extract_best_solution_filename(parsed_response["best_solution"])
    
    return {
        "parsed_response": parsed_response,
        "best_solution": best_solution_filename,
    }


# Usage example
if __name__ == "__main__":
    import yaml
    from dotenv import load_dotenv
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    
    with open("configs/rubrics.yaml", 'r', encoding='utf-8') as f:
        agent_config = yaml.safe_load(f)

    load_dotenv()
    load_dotenv(".local.env", override=True)

    model_name = agent_config["step_models"]["RubricsAgent"]["name"]
    model_api_config = agent_config["api_configs"][model_name]
    API_BASE = os.getenv(model_api_config["api_base"], model_api_config["api_base"])
    API_KEY = os.getenv(model_api_config["api_key_env"], model_api_config["api_key_env"])
    MODEL_NAME = model_name

    solution_top_k = agent_config["step_models"]["RubricsAgent"]["solution_top_k"]
    solution_path = agent_config["paths"]["solution_path"]
    already_processed_path = agent_config["logging"]["already_processed_path"]

    # 读取已处理的任务
    if os.path.exists(already_processed_path):
        with open(already_processed_path, 'r', encoding='utf-8') as f:
            already_processed = set([json.loads(line)["id"] for line in f])
    else:
        already_processed = set()
    
    # 创建锁用于文件写入
    file_lock = Lock()
    
    def process_task(specific_task):
        """处理单个任务的函数"""
        try:
            logger.info(msg=f"Processing folder: {specific_task}")
            
            # Execute comparison
            result = compare_trajectories(
                folder_path=os.path.join(solution_path, specific_task),
                solution_top_k = solution_top_k,
                api_base=API_BASE,
                api_key=API_KEY,
                model_name=MODEL_NAME,
            )
            
            if result:
                rubrics_output_path = os.path.join(solution_path, specific_task, "rubrics_output.json")
                with open(rubrics_output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)
                
                # 使用锁保护文件写入
                with file_lock:
                    with open(already_processed_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"id": specific_task}) + '\n')
                                
                return specific_task, True, None
            else:
                return specific_task, False, "No result returned"
                
        except Exception as e:
            logger.error(f"Error processing task {specific_task}: {str(e)}")
            return specific_task, False, str(e)
    
    # 获取待处理的任务列表
    tasks_to_process = [
        task for task in os.listdir(solution_path) 
        if task not in already_processed 
        and os.path.isdir(os.path.join(solution_path, task))
        and len([f for f in os.listdir(os.path.join(solution_path, task)) 
                if f.startswith("solution") and f.endswith(".json")]) > 1
    ]
    logger.info(f"Total tasks to process: {len(tasks_to_process)}")
    logger.info(f"Already processed: {len(already_processed)}")
    
    # 设置线程数（可以根据需要调整）
    max_workers = agent_config["processing"]["max_workers"]
    
    # 使用线程池执行任务
    success_count = 0
    failure_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(process_task, task): task 
            for task in tasks_to_process
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                task_id, success, error = future.result()
                if success:
                    success_count += 1
                    logger.info(f"✓ Task {task_id} completed successfully ({success_count}/{len(tasks_to_process)})")
                else:
                    failure_count += 1
                    logger.warning(f"✗ Task {task_id} failed: {error} ({failure_count} failures)")
            except Exception as e:
                failure_count += 1
                logger.error(f"✗ Task {task} raised an exception: {str(e)}")
    
    # 输出统计信息
    logger.info("="*50)
    logger.info(f"Processing complete!")
    logger.info(f"Total tasks: {len(tasks_to_process)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {failure_count}")
    logger.info("="*50)
