import re
from functions.call_llms import call_llm_api

tool_set_prompt = """
You are an expert at using multiple real-world tools creatively, and developing virtual tools that are feasible in real-world scenarios.
Task: Based on the given context, design a complex and meaningful plan that requires at least two different tools and ensures real-world feasibility. Additionally, conceive a virtual tool needed to accomplish the task — the tool should have simple and practical functions and should not undertake overly complex operations (for example, it may be used to research literature related to a physics problem, but should not directly provide the solution). Finally, outline and present the complete workflow.

Background information: {background_info}

Complexity Requirements
1. The task must involve information collection, processing, or creative production, and require 3--5 tools of different categories working together.
2. Tools must be realistic (you may assume hypothetical tools, but they must be based on existing technologies). No "all-powerful" or supernatural tools.
3. The whole task must be completable in a reasonable timeframe — not days/weeks of massive data gathering.
4. Each tool call must have clearly defined input/output schemas (with field names, data types, and descriptions).
5. The complete workflow should have 5-7 macro-level sub-goals necessary for task completion. Note that these goals should be high-level and do not require specifying tool call parameters.

You must follow these reasoning steps:

1. Background Analysis
   - Analyze the potential goals/motivations, list possible, achievable task directions

2. Task Definition
   - Choose one complex but feasible creative task, describe the task scenario
   - Additionally, you must generate one extra restriction that makes the task more challenging. This restriction must explicitly specify which tool cannot be used for what specific operation during the task execution. The forbidden tool usage must be clear, testable, and relevant to the task context. Violating this restriction would cause the entire task to fail. The restriction must follow this format: "Tool [tool_name] must NOT be used to [specific action/operation]."


3. Virtual Tool List Design
   For each tool: provide name / description / input parameters / required parameter_name / output format.
   The format must strictly match this style:

{{
    "name": "tool_name",
    "description": "Brief description of the tool functionality.",
    "parameters": {{
        "type": "object",
        "properties": {{
            "parameter_name": {{
                "description": "Description of the parameter",
                "type": "string"
            }}
            // Additional input parameters can be added here
        }},
        "required": ["parameter_name"]
    }},
    "outputs": {{
        "type": "object",
        "properties": {{
            "output_field_name": {{
                "description": "Description of the output field",
                "type": "string"
            }}
            // Additional output fields can be added here
        }}
    }}
}}

4. Complete Task Workflow
   - Define 5-7 macro-level sub-goals necessary for completion
   - Each sub-goal represents key milestones, not detailed operations
   - Multiple exploration paths are allowed to achieve sub-goals


Final Output Format:

<reasoning>(Step-by-step reasoning process…)</reasoning>

## 1. Possible complex action
<task>(Task description)</task>

## 2. Tool input/output definition (JSON)
<tools>(JSON schema)</tools>

## 3. Extra restriction with tool calls
<restriction>Tool [specific_tool_name] must NOT be used to [specific forbidden action/operation]. This restriction exists because [concrete real-world reason, e.g., maintenance downtime, budget limits, API deprecation, security concerns, or performance issues]. The [specific_tool_name] must be one of the tools defined in the "Tool input/output definition" section above. Violating this restriction will result in task failure.</restriction>

## 4. High-level Complete task flow
<workflow>(High-level detailed steps, it must follow the restriction above. Do not require specifying tool call parameters.)</workflow>
"""

def generate_tool_set(cfg, background_info):
    prompt = tool_set_prompt.format(background_info=background_info)
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

    # exclude content between <reasoning> and </reasoning>
    all_content = re.sub(r"<reasoning>(.+?)</reasoning>", "", all_content, flags=re.DOTALL)
    
    tool_matches = re.findall(r"<tools>(.+?)</tools>", all_content, re.DOTALL)
    if tool_matches:
        last_match = tool_matches[-1]
        tools = last_match.strip()
    else:
        tools = None
    
    workflow_matches = re.findall(r"<workflow>(.+?)</workflow>", all_content, re.DOTALL)
    if workflow_matches:
        last_match = workflow_matches[-1]
        workflow = last_match.strip()
    else:
        workflow = None

    task_matches = re.findall(r"<task>(.+?)</task>", all_content, re.DOTALL)
    if task_matches:
        last_match = task_matches[-1]
        task = last_match.strip()
    else:
        task = None

    
    restrict_matches = re.findall(r"<restriction>(.+?)</restriction>", all_content, re.DOTALL)
    if restrict_matches:
        last_match = restrict_matches[-1]
        restrict = last_match.strip()
    else:
        restrict = None

    return all_content, task, tools, workflow, restrict
