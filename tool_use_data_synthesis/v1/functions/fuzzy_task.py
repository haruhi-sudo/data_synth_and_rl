import re
from functions.call_llms import call_llm_api

fuzzy_task_prompt = """
You are an expert at using multiple real-world tools creatively, and developing virtual tools that are feasible in real-world scenarios.
Task: 

You are provided with: a task description, a set of virtual tools that may be helpful in completing the task, and an associated tool invocation workflow. Based on this information, construct a realistically feasible task to evaluate an intelligent agent’s ability to solve practical problems.


Output Requirements:
1. First, provide the step-by-step reasoning process you followed to design a high-quality, realistically feasible task. Wrap this in <reasoning></reasoning> tags. The reasoning must include your step-by-step thought process, explaining how you went from the given tools and workflow information to a compliant task description.
<reasoning>
(Step-by-step reasoning process…)
</reasoning>

2. Then, provide the task description, wrapped in <task></task> tags, it should not be too detailed, more information is presented in the next step:
<task>
(One‑sentence fuzzy task, it should be concise, and more information is provided in the next step)
</task>

3. Finally,provide additional task-related background information, including what must be understood to complete the task and key subtasks. Wrap this in tags:
<background>
(Background information Agent should know, it should be detailed)
</background>

The task and background description must:
- Not lengthy or too detailed.
- Avoid explicitly or implicitly instructing which tools to use. It's very important!
- Not directly provide the specific execution workflow.
- Have a clear and specific intent, allowing the intelligent agent to infer how to implement it.
- Include enough rich details so that the intelligent agent can naturally infer the need for multi-step, multi-tool integration.

Task short description, virtual tools, and workflow:
{initial_task_info}
"""

# - Be entirely virtual, with no connection to real people or events.
# Important Notes:
# - Because the tools are virtual and simulated by a large language model, if the task involves real people or events, it may lead to discrepancies between tool output and reality. Therefore, when constructing the task, use virtual characters, events, and locations.
# - When generating any virtual entities, always prefix their names with the word 'virtual', e.g., 'Virtual band Windhowl', 'Virtual city Aurora Falls'.

def generate_fuzzy_task(cfg, initial_task_info):
    prompt = fuzzy_task_prompt.format(initial_task_info=initial_task_info)
    messages = call_llm_api(
        user_prompt=prompt,
        system_prompt="",
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        model_name=cfg.model_name,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
    
    all_content = messages[-1]["content"]

    task_matches = re.findall(r"<task>(.+?)</task>", all_content, re.DOTALL)
    if task_matches:
        last_match = task_matches[-1]
        task = last_match.strip()
    else:
        task = None

    bg_matches = re.findall(r"<background>(.+?)</background>", all_content, re.DOTALL)
    if bg_matches:
        last_match = bg_matches[-1]
        bg = last_match.strip()
    else:
        bg = None

    return task, bg

