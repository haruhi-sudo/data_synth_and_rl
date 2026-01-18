import re
from .call_llms import call_llm_api

tool_set_prompt = """You are an expert capable of creating complex tool-use tasks and designing executable strategic rules.

Task Objective:  
Based on the provided background information, design a complex but realistically feasible task that requires at least two different tools working together.  
You must also design several virtual tools and define their input/output schemas.  
Finally, produce a tree-based policy that describes the agent's allowed behaviors, prohibited actions, tool preconditions, and refusal rules across different scenarios.

Notes:  
This task is *not* about designing a “guaranteed executable workflow,” but about designing a “policy tree with multiple scenario branches, constraints, and conditional logic.”  
In some scenarios, the agent should be able to execute the task; in others, it must refuse, avoid tool invocation, or follow strict tool‑order requirements.

Complexity Requirements:

1. The task must involve information gathering, processing, or creative generation, and require 3–5 different categories of tools working together.

2. Tools must be based on real-world technologies. They may be hypothetical virtual tools, but their functionality must be clear and limited in scope.

3. Tools must include explicit input/output structures (field names, types, descriptions).

4. You must construct a *tree‑based policy* for the task, including:
   - allowed actions (operations allowed when certain conditions are met)
   - disallowed actions
   - tool preconditions (requirements before a tool can be used)
   - refusal-required conditions
   - transfer-required conditions (when escalation to a human is necessary)
   - cross-policy dependencies

Note that you should not impose any restrictions on the order of task completion process,  or which tools must be called.

5. The policy must be presented as a JSON tree containing multiple scenario branches, describing decision paths for the agent.

----------------------------------------
Reasoning Steps:

1. Background Analysis  
   - Infer possible user goals based on the background  
   - List 2–3 feasible high-level task directions  
   - Choose the most complex one suited for multi-tool collaboration

2. Task Definition  
   - Describe the complex task: its goal, context, and key challenges

3. Tool List Design  
Design 3–5 tools.  
Each tool must include: name, description, parameters, outputs using this format:

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

4. Tree-based Policy (Core Part)  
Produce a JSON policy tree containing:

- "root_condition": top-level trigger condition  
- "allowed_actions"  
- "disallowed_actions"  
- "tool_preconditions"  
- "refusal_conditions"  
- "transfer_conditions"  
- "branches": a recursive JSON tree of scenario branches  
  Each branch must have:  
  - "condition"  
  - "next" (sub-branches)  
  - "outcome" (allowed / denied / ask for clarification)
- other actions that can make the task more complex

The policy does not need to cover every real-world scenario but must be sufficiently complex and multi-layered.

{{
  "root_condition": "User requests an analytical briefing regarding the political or constitutional legitimacy of a specified election event.",

  "allowed_actions": [
    "Use legal_doc_retriever for constitutional or judicial materials",
    "Use news_aggregator on vetted and neutral news categories",
    "Use sentiment_analyzer with geographic+language filters",
    "Use fact_checker on specific factual claims",
    "Ask clarifying questions when key details are missing",
    "Synthesize information into a neutral, sourced summary"
  ],

  "disallowed_actions": [
    "Make normative or moral judgments",
    "Predict future political events",
    "Cite unverified or anonymous sources",
    "Use tools with missing preconditions",
    "Invoke tools in the wrong sequence"
  ],

  "clarification_required": [
    "Missing event date",
    "Missing jurisdiction",
    "User mixes unrelated countries/events"
  ],

  "tool_preconditions": {{
    "legal_doc_retriever": {{ "must_have": ["valid_date_range", "jurisdiction"] }},
    "news_aggregator": {{ "must_have": ["approved_outlet_types"] }},
    "sentiment_analyzer": {{ "must_have": ["geo_filter", "language_filter"] }},
    "fact_checker": {{ "must_have": ["falsifiable_claim"] }}
  }},

  "refusal_conditions": [
    "User asks for normative judgment",
    "User presents conspiracy theories",
    "User requests real‑time polling data"
  ],

  "transfer_conditions": [
    "Legal documents conflict without clear resolution",
    "Fact-checker repeatedly returns 'unverifiable'",
    "Sentiment analysis shows extreme geographic or demographic bias"
  ],

  "branches": [

    {{
      "condition": "User explicitly requests a normative judgment (e.g., moral legitimacy)",
      "action": "refuse",
      "next": null
    }},

    {{
      "condition": "User query missing jurisdiction or event date",
      "action": "clarify",
      "next": null
    }},

    {{
      "condition": "User provides clear factual request with jurisdiction + date specified",
      "action": "proceed",
      "next": [
        {{
          "condition": "Legal documents retrieved successfully",
          "action": "continue_sequence",
          "next": null
        }},
        {{
          "condition": "Legal retrieval fails or contradicts itself",
          "action": "transfer",
          "next": null
        }}
      ]
    }},

    {{
      "condition": "User mixes unrelated geopolitical events (e.g., references two countries' elections together)",
      "action": "clarify",
      "next": [
        {{
          "condition": "User provides corrected and consistent scope",
          "action": "proceed",
          "next": null
        }},
        {{
          "condition": "User refuses to clarify scope",
          "action": "refuse",
          "next": null
        }}
      ]
    }},

    {{
      "condition": "User provides complete event details + valid jurisdiction",
      "action": "proceed",
      "next": [
        {{
          "condition": "Legal documents are consistent and meet preconditions",
          "action": "continue_sequence",
          "next": [
            {{
              "condition": "News aggregator produces coherent multi-source coverage",
              "action": "continue_sequence",
              "next": [
                {{
                  "condition": "Sentiment analysis shows balanced geographic signals",
                  "action": "allowed",
                  "next": null
                }},
                {{
                  "condition": "Sentiment analysis shows severe bias",
                  "action": "refuse",
                  "next": null
                }}
              ]
            }},
            {{
              "condition": "News sources inconsistent or flagged unreliable",
              "action": "refuse",
              "next": null
            }}
          ]
        }},
        {{
          "condition": "Legal documents conflicting or missing",
          "action": "transfer",
          "next": null
        }}
      ]
    }}
  ]
}}

----------------------------------------
Background Information: {background_info}
----------------------------------------

----------------------------------------
Final Output Format (must follow strictly):

<reasoning>step-by-step reasoning…</reasoning>

## 1. Task Description
<task>(task description)</task>

## 2. Tool List (JSON)
<tools>(tools JSON list)</tools>

## 3. Tree-based Policy (JSON)
<policy_tree>(policy tree JSON)</policy_tree>
"""

def generate_tool_set_policy(cfg, background_info):
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
    
    policy_matches = re.findall(r"<policy_tree>(.+?)</policy_tree>", all_content, re.DOTALL)
    if policy_matches:
        last_match = policy_matches[-1]
        policy = last_match.strip()
    else:
        policy = None

    task_matches = re.findall(r"<task>(.+?)</task>", all_content, re.DOTALL)
    if task_matches:
        last_match = task_matches[-1]
        task = last_match.strip()
    else:
        task = None

    return all_content, task, tools, policy

if __name__ == "__main__":
    import yaml
    import json
    from langchain_core.runnables import RunnableConfig
    from configuration import ModelConfiguration

    with open("configs/tool_use_data_gen.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    task_bg = "A maternal health advocate focused on raising awareness about postpartum complications."
    def create_step_config(
        base_config: RunnableConfig, step_name: str, 
    ) -> RunnableConfig:
        """Create a new configuration for a specific step with its designated model"""
        # cfg = AgentConfiguration.from_runnable_config(base_config)
        step_model_config = base_config["step_models"][step_name]
        
        # Create a new config with the specific model for this step
        step_config = {}
        if "configurable" not in step_config:
            step_config["configurable"] = {}
            
        # Apply the step-specific model configuration
        step_config["configurable"]["model_name"] = step_model_config["name"]
        if "temperature" in step_model_config:
            step_config["configurable"]["temperature"] = step_model_config["temperature"]
        if "max_tokens" in step_model_config:
            step_config["configurable"]["max_tokens"] = step_model_config["max_tokens"]
        if "use_tools" in step_model_config:
            step_config["configurable"]["use_tools"] = step_model_config["use_tools"]
        if "use_thinking" in step_model_config:
            step_config["configurable"]["use_thinking"] = step_model_config["use_thinking"]

        step_config["configurable"]["api_configs"] = base_config["api_configs"]
        
        return step_config

    with open("configs/tool_use_data_gen.yaml", 'r', encoding='utf-8') as f:
        agent_config = yaml.safe_load(f)
    
    step_config = create_step_config(agent_config, "ToolSetGenAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)

    generate_tool_set(cfg, task_bg)

