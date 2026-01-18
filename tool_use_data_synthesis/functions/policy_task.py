import re
from .call_llms import call_llm_api

policy_task_prompt = """
You are an expert at transforming structured Tree-based Policies (JSON format) into human‑readable natural‑language manuals, and generating test cases derived from such policies.

You will receive three inputs: task_description, policy_tree, and tools. Your output contains two major sections.

============================================================
# Part 1: Produce a Natural-Language Policy
============================================================

Generate a detailed natural‑language policy document based on the JSON policy.  

Guidelines:
- Do NOT copy the JSON item by item. Convert it into a coherent human-readable policy document.
- To increase the difficulty of the task, you cannot directly explain the workflow. You cannot tell the agent what to do, only what not to do. And do not specify the tool calls.
- Focus especially on:
  * Disallowed actions (write them clearly)
  * Refusal conditions (when the assistant must refuse)
  * Transfer conditions (when human intervention is mandatory)
  * Clarification-required conditions (when the system must ask the user questions)
  * Tool preconditions (what must be true before each tool is called)
  * Branch logic (explain, in plain language, how the system behaves in different cases)

Note that you should not impose any restrictions on the order of task completion process,  or which tools must be called.

The final result should read like a formal “Policies and Procedural Guidelines” document — not like JSON, not like a bullet dump.

Example:
```
# Retail agent policy\n\nAs a retail agent, you can help users:\n\n- **cancel or modify pending orders**\n- **return or exchange delivered orders**\n- **modify their default user address**\n- **provide information about their own profile, orders, and related products**\n\nAt the beginning of the conversation, you have to authenticate the user identity by locating their user id via email, or via name + zip code. This has to be done even when the user already provides the user id.\n\nOnce the user has been authenticated, you can provide the user with information about order, product, profile information, e.g. help the user look up order id.\n\nYou can only help one user per conversation (but you can handle multiple requests from the same user), and must deny any requests for tasks related to any other user.\n\nBefore taking any action that updates the database (cancel, modify, return, exchange), you must list the action details and obtain explicit user confirmation (yes) to proceed.\n\nYou should not make up any information or knowledge or procedures not provided by the user or the tools, or give subjective recommendations or comments.\n\nYou should at most make one tool call at a time, and if you take a tool call, you should not respond to the user at the same time. If you respond to the user, you should not make a tool call at the same time.\n\nYou should deny user requests that are against this policy.\n\nYou should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions. To transfer, first make a tool call to transfer_to_human_agents, and then send the message 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.' to the user.\n\n## Domain basic\n\n- All times in the database are EST and 24 hour based. For example \"02:30:00\" means 2:30 AM EST.\n\n### User\n\nEach user has a profile containing:\n\n- unique user id\n- email\n- default address\n- payment methods.\n\nThere are three types of payment methods: **gift card**, **paypal account**, **credit card**.\n\n### Product\n\nOur retail store has 50 types of products.\n\nFor each **type of product**, there are **variant items** of different **options**.\n\nFor example, for a 't-shirt' product, there could be a variant item with option 'color blue size M', and another variant item with option 'color red size L'.\n\nEach product has the following attributes:\n\n- unique product id\n- name\n- list of variants\n\nEach variant item has the following attributes:\n\n- unique item id\n- information about the value of the product options for this item.\n- availability\n- price\n\nNote: Product ID and Item ID have no relations and should not be confused!\n\n### Order\n\nEach order has the following attributes:\n\n- unique order id\n- user id\n- address\n- items ordered\n- status\n- fullfilments info (tracking id and item ids)\n- payment history\n\nThe status of an order can be: **pending**, **processed**, **delivered**, or **cancelled**.\n\nOrders can have other optional attributes based on the actions that have been taken (cancellation reason, which items have been exchanged, what was the exchane price difference etc)\n\n## Generic action rules\n\nGenerally, you can only take action on pending or delivered orders.\n\nExchange or modify order tools can only be called once per order. Be sure that all items to be changed are collected into a list before making the tool call!!!\n\n## Cancel pending order\n\nAn order can only be cancelled if its status is 'pending', and you should check its status before taking the action.\n\nThe user needs to confirm the order id and the reason (either 'no longer needed' or 'ordered by mistake') for cancellation. Other reasons are not acceptable.\n\nAfter user confirmation, the order status will be changed to 'cancelled', and the total will be refunded via the original payment method immediately if it is gift card, otherwise in 5 to 7 business days.\n\n## Modify pending order\n\nAn order can only be modified if its status is 'pending', and you should check its status before taking the action.\n\nFor a pending order, you can take actions to modify its shipping address, payment method, or product item options, but nothing else.\n\n### Modify payment\n\nThe user can only choose a single payment method different from the original payment method.\n\nIf the user wants the modify the payment method to gift card, it must have enough balance to cover the total amount.\n\nAfter user confirmation, the order status will be kept as 'pending'. The original payment method will be refunded immediately if it is a gift card, otherwise it will be refunded within 5 to 7 business days.\n\n### Modify items\n\nThis action can only be called once, and will change the order status to 'pending (items modifed)'. The agent will not be able to modify or cancel the order anymore. So you must confirm all the details are correct and be cautious before taking this action. In particular, remember to remind the customer to confirm they have provided all the items they want to modify.\n\nFor a pending order, each item can be modified to an available new item of the same product but of different product option. There cannot be any change of product types, e.g. modify shirt to shoe.\n\nThe user must provide a payment method to pay or receive refund of the price difference. If the user provides a gift card, it must have enough balance to cover the price difference.\n\n## Return delivered order\n\nAn order can only be returned if its status is 'delivered', and you should check its status before taking the action.\n\nThe user needs to confirm the order id and the list of items to be returned.\n\nThe user needs to provide a payment method to receive the refund.\n\nThe refund must either go to the original payment method, or an existing gift card.\n\nAfter user confirmation, the order status will be changed to 'return requested', and the user will receive an email regarding how to return items.\n\n## Exchange delivered order\n\nAn order can only be exchanged if its status is 'delivered', and you should check its status before taking the action. In particular, remember to remind the customer to confirm they have provided all items to be exchanged.\n\nFor a delivered order, each item can be exchanged to an available new item of the same product but of different product option. There cannot be any change of product types, e.g. modify shirt to shoe.\n\nThe user must provide a payment method to pay or receive refund of the price difference. If the user provides a gift card, it must have enough balance to cover the price difference.\n\nAfter user confirmation, the order status will be changed to 'exchange requested', and the user will receive an email regarding how to return items. There is no need to place a new order.
```

============================================================
# Part 2: Generate Test Cases
============================================================

Produce multiple test cases demonstrating how the policy behaves in different scenarios.  

Required scenario coverage
1. Fully valid request (should execute successfully)
2. A request that must be refused (from refusal conditions) or clarification or violating tool_preconditions
3. A mixed or complex case (e.g., user insists on skipping mandatory verification)

You may add additional representative cases if helpful.

Example:
```
Policy: Keyboard replacement is allowed, but not other products.

Task: Hi, I want to exchange my order #W2378156 for a new keyboard.

User Background Information: You received your order #W2378156 and wish to exchange the mechanical keyboard for a similar one, but with clicky switches and a smart thermostat compatible with Google Home instead of Apple HomeKit. If there is no keyboard with clicky switches, RGB backlight, and full size, you'd rather only exchange the thermostat.

Scenarios that can be tested include: Does the agent refuse a user's request to exchange the keyboard for a thermostat?
```

============================================================
# Final Output Format (must follow exactly)
============================================================

<natural_language_policy>
(Your Part 1 output)，Note that you should not impose any restrictions on the order of task completion process, or which tools must be called.
</natural_language_policy>

<test_case>
<task>...</task>
<user_background>...</user_background>
<test_policy>...</test_policy>
</test_case>
You can repeat this for multiple test cases.

============================================================
INPUTS:
============================================================

1. <task_description>
{task_description}
</task_description>

2. <policy_tree>
{policy_tree}
</policy_tree>

3. <tools>
{tools}
</tools>
"""


def generate_policy_test_case(cfg, task_description, policy_tree, tools):
    prompt = policy_task_prompt.format(task_description=task_description, policy_tree=policy_tree, tools=tools)
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

    policy_matches = re.findall(r"<natural_language_policy>(.+?)</natural_language_policy>", all_content, re.DOTALL)
    if policy_matches:
        last_match = policy_matches[-1]
        policy = last_match.strip()
    else:
        policy = None

    test_case_matches = re.findall(r"<test_case>(.+?)</test_case>", all_content, re.DOTALL)

    # multiple test cases
    test_cases_task = []
    test_cases_user_bg = []
    for test_case_match in test_case_matches:
        test_case = test_case_match.strip()
        test_task_matches = re.findall(r"<task>(.+?)</task>", test_case, re.DOTALL)
        if test_task_matches:
            last_match = test_task_matches[-1]
            test_task = last_match.strip()
        else:
            test_task = None
        test_user_background_matches = re.findall(r"<user_background>(.+?)</user_background>", test_case, re.DOTALL)
        if test_user_background_matches:
            last_match = test_user_background_matches[-1]
            test_user_background = last_match.strip()
        else:
            test_user_background = None
        test_test_policy_matches = re.findall(r"<test_policy>(.+?)</test_policy>", test_case, re.DOTALL)
        if test_test_policy_matches:
            last_match = test_test_policy_matches[-1]
            test_test_policy = last_match.strip()
        else:
            test_test_policy = None
        
        if test_task and test_user_background:
            test_cases_task.append(test_task)
            test_cases_user_bg.append(test_user_background)
    
    return all_content, policy, test_case_matches

if __name__ == "__main__":
    import yaml
    import json
    from langchain_core.runnables import RunnableConfig
    from configuration import ModelConfiguration

    with open("configs/tool_use_data_gen.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    task_description = 'Generate a culturally appropriate, evidence-based policy brief on postpartum complications in a specified geographic region, targeting either healthcare providers, policymakers, or community educators. The brief must integrate recent clinical findings, public health data, demographic risk factors, and culturally relevant communication strategies. The agent must ensure all sources are current, data is regionally accurate, and language is adapted to the intended audience without perpetuating stereotypes or misinformation.'
    policy_tree = '{\n  "root_condition": "User requests a policy brief, educational material, or awareness report on postpartum complications in a specific region.",\n\n  "allowed_actions": [\n    "Use medical_research_retriever for clinical evidence",\n    "Use public_health_data_aggregator for official statistics",\n    "Use demographic_risk_analyzer to assess disparities",\n    "Use multilingual_summarizer to produce accessible content",\n    "Use cultural_validator to screen for sensitivity",\n    "Ask clarifying questions if region, audience, or condition is unspecified",\n    "Synthesize findings into a structured, sourced brief"\n  ],\n\n  "disallowed_actions": [\n    "Generate content without verified data sources",\n    "Produce summaries in unsupported languages",\n    "Skip cultural validation when targeting vulnerable communities",\n    "Use outdated data (older than 5 years) without disclaimer",\n    "Make causal claims not supported by evidence"\n  ],\n\n  "clarification_required": [\n    "Missing geographic region",\n    "Unspecified target audience",\n    "Undefined maternal complication type",\n    "Ambiguous language or mixed cultural references"\n  ],\n\n  "tool_preconditions": {\n    "medical_research_retriever": {\n      "must_have": ["condition", "geographic_focus", "publication_date_range"]\n    },\n    "public_health_data_aggregator": {\n      "must_have": ["region", "indicator_types", "year_range"]\n    },\n    "demographic_risk_analyzer": {\n      "must_have": ["region", "risk_factors"]\n    },\n    "multilingual_summarizer": {\n      "must_have": ["source_text", "target_language", "audience_type", "max_word_count"]\n    },\n    "cultural_validator": {\n      "must_have": ["text_to_evaluate", "target_culture"]\n    }\n  },\n\n  "refusal_conditions": [\n    "Request promotes harmful stereotypes",\n    "User demands omission of critical risk data",\n    "Target region has active conflict or data embargo"\n  ],\n\n  "transfer_conditions": [\n    "Cultural validator returns \'fail\' twice after revision",\n    "Public health data quality flag is \'low\' and no alternative source exists",\n    "Demographic risk analysis shows >3x disparity without explanatory context",\n    "Medical research confidence score < 0.5 across all studies"\n  ],\n\n  "branches": [\n    {\n      "condition": "User request lacks geographic region or specifies multiple conflicting regions",\n      "action": "clarify",\n      "next": null\n    },\n    {\n      "condition": "User asks for normative statements (e.g., blaming cultural practices)",\n      "action": "refuse",\n      "next": null\n    },\n    {\n      "condition": "User mixes unrelated health topics (e.g., postpartum depression and infant nutrition) without scope definition",\n      "action": "clarify",\n      "next": [\n        {\n          "condition": "User narrows focus to one condition and region",\n          "action": "proceed",\n          "next": null\n        },\n        {\n          "condition": "User insists on broad, unfocused output",\n          "action": "refuse",\n          "next": null\n        }\n      ]\n    },\n    {\n      "condition": "User provides complete inputs: region, condition, audience, and timeframe",\n      "action": "proceed",\n      "next": [\n        {\n          "condition": "Medical research retriever returns confidence_score >= 0.6",\n          "action": "continue_sequence",\n          "next": [\n            {\n              "condition": "Public health data quality flag is \'high\' or \'medium\'",\n              "action": "continue_sequence",\n              "next": [\n                {\n                  "condition": "Demographic risk analysis completes with confidence_interval provided",\n                  "action": "continue_sequence",\n                  "next": [\n                    {\n                      "condition": "Multilingual summarizer generates content",\n                      "action": "continue_sequence",\n                      "next": [\n                        {\n                          "condition": "Cultural validator returns \'pass\'",\n                          "action": "allowed",\n                          "next": null\n                        },\n                        {\n                          "condition": "Cultural validator returns \'warning\'",\n                          "action": "revise_and_recheck",\n                          "next": [\n                            {\n                              "condition": "Second validation passes",\n                              "action": "allowed",\n                              "next": null\n                            },\n                            {\n                              "condition": "Second validation fails",\n                              "action": "transfer",\n                              "next": null\n                            }\n                          ]\n                        },\n                        {\n                          "condition": "Cultural validator returns \'fail\'",\n                          "action": "transfer",\n                          "next": null\n                        }\n                      ]\n                    }\n                  ]\n                },\n                {\n                  "condition": "Demographic risk analysis fails or returns incomplete drivers",\n                  "action": "transfer",\n                  "next": null\n                }\n              ]\n            },\n            {\n              "condition": "Public health data quality flag is \'low\'",\n              "action": "transfer",\n              "next": null\n            }\n          ]\n        },\n        {\n          "condition": "Medical research confidence_score < 0.5",\n          "action": "transfer",\n          "next": null\n        }\n      ]\n    },\n    {\n      "condition": "User requests dissemination of unvalidated draft to public channels",\n      "action": "refuse",\n      "next": [\n        {\n          "condition": "Content has passed cultural validation and data verification",\n          "action": "allow_export_with_disclaimer",\n          "next": null\n        },\n        {\n          "condition": "Any tool step was skipped or failed",\n          "action": "deny_distribution",\n          "next": null\n        }\n      ]\n    }\n  ]\n}'
    tools = '[\n  {\n    "name": "medical_research_retriever",\n    "description": "Searches peer-reviewed medical literature databases for studies related to postpartum complications within a specified date range and geographic focus.",\n    "parameters": {\n      "type": "object",\n      "properties": {\n        "condition": {\n          "description": "Medical condition of interest (e.g., postpartum hemorrhage, preeclampsia)",\n          "type": "string"\n        },\n        "geographic_focus": {\n          "description": "Region or country of relevance for study applicability",\n          "type": "string"\n        },\n        "publication_date_range": {\n          "description": "Start and end dates for publication years in YYYY-MM format",\n          "type": "string"\n        }\n      },\n      "required": ["condition", "geographic_focus", "publication_date_range"]\n    },\n    "outputs": {\n      "type": "object",\n      "properties": {\n        "studies_found": {\n          "description": "List of relevant peer-reviewed studies with titles, authors, journals, and key findings",\n          "type": "array",\n          "items": {\n            "type": "object",\n            "properties": {\n              "title": { "type": "string" },\n              "authors": { "type": "array", "items": { "type": "string" } },\n              "journal": { "type": "string" },\n              "publication_date": { "type": "string" },\n              "key_findings": { "type": "array", "items": { "type": "string" } }\n            }\n          }\n        },\n        "confidence_score": {\n          "description": "Aggregated reliability score from 0 to 1 based on sample size, study design, and citation count",\n          "type": "number"\n        }\n      }\n    }\n  },\n  {\n    "name": "public_health_data_aggregator",\n    "description": "Retrieves official maternal health statistics including mortality rates, facility access, and antenatal care coverage from trusted national or international health organizations.",\n    "parameters": {\n      "type": "object",\n      "properties": {\n        "region": {\n          "description": "Specific country or subnational region (e.g., state, province)",\n          "type": "string"\n        },\n        "indicator_types": {\n          "description": "Types of indicators requested (e.g., maternal_mortality_ratio, postnatal_care_coverage)",\n          "type": "array",\n          "items": { "type": "string" }\n        },\n        "year_range": {\n          "description": "Years for which data is requested",\n          "type": "string"\n        }\n      },\n      "required": ["region", "indicator_types", "year_range"]\n    },\n    "outputs": {\n      "type": "object",\n      "properties": {\n        "data_records": {\n          "description": "Key maternal health indicators with values, units, and source metadata",\n          "type": "array",\n          "items": {\n            "type": "object",\n            "properties": {\n              "indicator": { "type": "string" },\n              "value": { "type": "number" },\n              "unit": { "type": "string" },\n              "source": { "type": "string" },\n              "year": { "type": "string" },\n              "geographic_level": { "type": "string" }\n            }\n          }\n        },\n        "data_quality_flag": {\n          "description": "Indicator of data completeness and timeliness: \'high\', \'medium\', \'low\'",\n          "type": "string"\n        }\n      }\n    }\n  },\n  {\n    "name": "demographic_risk_analyzer",\n    "description": "Models relative maternal risk levels based on socioeconomic, geographic, and healthcare access variables within a region.",\n    "parameters": {\n      "type": "object",\n      "properties": {\n        "region": {\n          "description": "Target region for risk modeling",\n          "type": "string"\n        },\n        "risk_factors": {\n          "description": "List of input factors (e.g., rural_residence, low_income, minority_ethnicity)",\n          "type": "array",\n          "items": { "type": "string" }\n        }\n      },\n      "required": ["region", "risk_factors"]\n    },\n    "outputs": {\n      "type": "object",\n      "properties": {\n        "risk_map": {\n          "description": "Estimated relative risk scores per demographic subgroup",\n          "type": "object",\n          "additionalProperties": { "type": "number" }\n        },\n        "primary_drivers": {\n          "description": "Top three factors contributing to elevated risk",\n          "type": "array",\n          "items": { "type": "string" }\n        },\n        "confidence_interval": {\n          "description": "Statistical confidence range for risk estimates",\n          "type": "string"\n        }\n      }\n    }\n  },\n  {\n    "name": "multilingual_summarizer",\n    "description": "Generates concise summaries of technical content in multiple languages, adapted to audience literacy level.",\n    "parameters": {\n      "type": "object",\n      "properties": {\n        "source_text": {\n          "description": "Original technical or scientific text to summarize",\n          "type": "string"\n        },\n        "target_language": {\n          "description": "Language code for output (e.g., es, hi, fr)",\n          "type": "string"\n        },\n        "audience_type": {\n          "description": "Intended reader group: clinician, policymaker, community_member",\n          "type": "string"\n        },\n        "max_word_count": {\n          "description": "Maximum length of summary",\n          "type": "integer"\n        }\n      },\n      "required": ["source_text", "target_language", "audience_type", "max_word_count"]\n    },\n    "outputs": {\n      "type": "object",\n      "properties": {\n        "summary_text": {\n          "description": "Generated summary in target language",\n          "type": "string"\n        },\n        "reading_level": {\n          "description": "Flesch-Kincaid grade level of the output",\n          "type": "number"\n        },\n        "key_messages": {\n          "description": "List of core takeaways preserved in summary",\n          "type": "array",\n          "items": { "type": "string" }\n        }\n      }\n    }\n  },\n  {\n    "name": "cultural_validator",\n    "description": "Analyzes text for potential cultural insensitivity, stigmatizing language, or inappropriate assumptions regarding gender, ethnicity, or tradition.",\n    "parameters": {\n      "type": "object",\n      "properties": {\n        "text_to_evaluate": {\n          "description": "Text segment requiring cultural review",\n          "type": "string"\n        },\n        "target_culture": {\n          "description": "Cultural context of the audience (e.g., Yoruba_Nigeria, Quechua_Peru)",\n          "type": "string"\n        },\n        "gender_sensitivity_required": {\n          "description": "Whether strict gender-neutral or gender-affirming language is required",\n          "type": "boolean"\n        }\n      },\n      "required": ["text_to_evaluate", "target_culture"]\n    },\n    "outputs": {\n      "type": "object",\n      "properties": {\n        "flagged_terms": {\n          "description": "List of words or phrases identified as potentially offensive or misleading",\n          "type": "array",\n          "items": { "type": "string" }\n        },\n        "context_warnings": {\n          "description": "Explanations of why certain terms may be problematic",\n          "type": "array",\n          "items": { "type": "string" }\n        },\n        "validation_status": {\n          "description": "Result of evaluation: \'pass\', \'warning\', \'fail\'",\n          "type": "string"\n        }\n      }\n    }\n  }\n]'
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

    generate_policy_test_case(cfg, task_description, policy_tree, tools)
