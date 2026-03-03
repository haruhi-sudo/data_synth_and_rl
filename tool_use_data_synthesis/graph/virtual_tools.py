import os
import json
import yaml
import glob
import re
import threading
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from configuration import ModelConfiguration
from functions.policy_task import generate_policy_test_case
from functions.tool_set_policy_gen import generate_tool_set_policy
from functions.refine_policy_task import generate_task_and_user_background

# Add a lock for thread-safe file writing
log_file_lock = threading.Lock()

class AgentState(TypedDict):
    seed_info: Dict[str, Any]  # To store original task information
    breaked: bool # It will be set to False when any processing step fails

    initial_toolset_create: str
    initial_tools: str
    initial_task: str
    initial_policy: str

    policy_str: str
    test_cases: List[str]

    checked_tools: List[Dict[str, Any]]
    tasks_and_backgrounds: List[Dict[str, Any]]


_STEP_CONFIG_KEYS = ("model_name", "temperature", "max_tokens")

def create_step_config(
    base_config: RunnableConfig, step_name: str,
) -> RunnableConfig:
    """Create a new configuration for a specific step with its designated model."""
    step_model = base_config["configurable"]["step_models"][step_name]
    configurable = {k: v for k, v in step_model.items() if k in _STEP_CONFIG_KEYS}
    configurable["api_configs"] = base_config["configurable"]["api_configs"]
    return {"configurable": configurable}

def toolset_gen_node(state: AgentState, config: RunnableConfig):
    # Create step-specific configuration
    step_config = create_step_config(config, "ToolSetGenAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)
    
    original_bg = state["seed_info"]["background"]
    all_content, task, tools, policy = generate_tool_set_policy(cfg=cfg, background_info=original_bg)

    if task is None or tools is None or policy is None:
        return {
            "breaked": True,
            "initial_toolset_create": None,
            "initial_task": None,
            "initial_tools": None,
            "initial_policy": None
        }

    return {
        "initial_toolset_create": all_content,
        "initial_task": task,
        "initial_tools": tools,
        "initial_policy": policy
    }

def policy_task_node(state: AgentState, config: RunnableConfig):
    if state["breaked"]:
        return {
            "policy_str": None,
            "test_cases": None
        }
    # Create step-specific configuration
    step_config = create_step_config(config, "PolicyTaskAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)
    
    task_description = state["initial_task"]
    policy_tree = state["initial_policy"]
    tools = state["initial_tools"]
    try:
        tools = json.loads(tools)
    except:
        print(f"Error: The tools is not a valid JSON string: {tools}")
        return {
            "breaked": True,
            "policy_str": None,
            "test_cases": None
        }
    all_content, policy, test_cases_task_bg_policy = generate_policy_test_case(cfg, task_description, policy_tree, tools)
    
    return {
        "checked_tools": tools,
        "policy_str": policy,
        "test_cases": test_cases_task_bg_policy
    }

def final_task_node(state: AgentState, config: RunnableConfig):
    if state["breaked"]:
        return {
            "tasks_and_backgrounds": None
        }
    # Create step-specific configuration
    step_config = create_step_config(config, "FinalTaskAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)

    test_cases = state["test_cases"]
    tools = state["checked_tools"]

    task_and_user_background = generate_task_and_user_background(cfg, tools, test_cases)

    if len(task_and_user_background) == 0 or task_and_user_background is None:
        return {
            "breaked": True,
            "tasks_and_backgrounds": None
        }
    
    return {
        "tasks_and_backgrounds": task_and_user_background
    }

# Build the graph
builder = StateGraph(AgentState, config_schema=RunnableConfig)
builder.add_node("toolset_gen", toolset_gen_node)
builder.add_node("policy_task", policy_task_node)
builder.add_node("final_task", final_task_node)

builder.set_entry_point("toolset_gen")
builder.add_edge("toolset_gen", "policy_task")
builder.add_edge("policy_task", "final_task")
graph = builder.compile()

# --- 运行入口 ---
def run_agent(seed_info: dict, run_config: dict = None):
    virtual_tool_use_task_path = run_config["logging"]["task_file_path"]
    
    log_dir = os.path.dirname(virtual_tool_use_task_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    run_config = {"configurable": run_config or {}}

    initial_state = {
        "seed_info": seed_info,
        "breaked": False,
    }
    run_config["recursion_limit"] = 100
    final_state = graph.invoke(initial_state, config=run_config)

    save_data = {
        "id": seed_info["id"],
        "checked_tools": final_state["checked_tools"],
        "policy": final_state["policy_str"],
        "tasks_and_backgrounds": final_state["tasks_and_backgrounds"],
    }

    if not final_state["breaked"]:
        with log_file_lock:
            # Save basic task data
            with open(virtual_tool_use_task_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(save_data, ensure_ascii=False) + '\n')
    
    return final_state


if __name__ == "__main__":
    with open("configs/data_gen.yaml", 'r', encoding='utf-8') as f:
        agent_config = yaml.safe_load(f)

    # Example usage
    task = {"id": "2e9f925f-0299-4255-82ed-b8bb2dc9627e", "background": "a passionate fan of Afrikaans music and die-hard supporter of Spoegwolf"}
    with open("configs/persona_5K.jsonl", 'r', encoding='utf-8') as f:
        tasks = [json.loads(line) for line in f]

    for task in tasks:
        new_task = {
            "id": task["id"],
            "background": task["persona"]
        }
        run_agent(new_task, run_config=agent_config)

