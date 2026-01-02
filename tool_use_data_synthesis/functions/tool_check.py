import re
from .call_llms import call_llm_api

tool_check_prompt = """
You are an expert at designing virtual tools that are feasible in real-world scenarios.
Now, there is a set of virtual tools that intelligent agents can call. Some of these tools may be designed in an unreasonable way. Your task is to review and modify them so that they are both reasonable and concise in a virtual environment.

Virtual tools are divided into two categories:
- State-modifying tools: Tools that change a certain state or perform an action. For example, turning on the phone's Wi-Fi.
- Content-returning tools: Tools that return certain information. For example, Web Search or Web Browser returning retrieved text information.

You must follow these rules:

1. If a tool's function can essentially be performed by the intelligent agent itself, it should not be designed as a separate tool.
   - For example, content_generator: converting structured data into high-quality explanations, summaries, or narrative content is something an LLM can already do without requiring a tool. Another exmaple: sentiment_analyzer, it can be done by the LLM itself. Remove such redundant tools directly.

2. Ensure tool outputs can be reasonably simulated by the LLM in a virtual context. The tool should have a real-world prototype.
   - These tools are virtual and completely mimic real-world tools.
   - The calls and return values of tools are simulated entirely by a text-based LLM in conversation, and will not be actually deployed. Therefore, the tool's output must be something that the LLM can reasonably produce in text, rather than requiring real multimodal data generation (e.g., real images, audio, or video files).
   - If a tool is originally designed to return non-textual, non-simulatable content, modify it to instead return a creation success status and, if necessary, auxiliary information (such as a virtual ID or textual description).

3. Reduce the number of returned parameters and keep tools simple
   - Each tool should return no more than two parameters whenever possible.
   - Each tool should have a single, clear purpose; it should not try to perform too many tasks or return too much information.

4. To increase task difficulty, add 1 to 2 additional tools that are related to the context but serve as distractors. These distractor tools should be plausible and satisfy Rule 1 and 2. But they are clearly unnecessary or unreasonable, so that they challenge the reviewing process.

5. Finally, make sure the task can be completed by these virtual tools.

Output Requirements:
1. First, provide the step-by-step reasoning process for reviewing each tool, including whether it is redundant, whether its output can be reasonably simulated by a purely text-based LLM, and any other thinking related to making virtual tools reasonable and concise. Wrap this in <reasoning> tags:
<reasoning>
(Step-by-step reasoning process…)
</reasoning>

2. Then, output the reviewed and modified tools that can be simulated by LLMs, wrapped in <tools> tags:
<tools>
(Virtual tools that can be simulated by LLMs)
</tools>

Important:  A valid tool JSON array should be parsed by `json.loads` into a `List[Dict[str, Any]]`. Note that the tool's name should conform to coding style guidelines. For example, `GeoMap_Studio` instead of `GeoMap Studio`.

Task Description: {task_description}

Tool Description: {tool_description}
"""


def tool_check(cfg, tool_description, task_description):
    prompt = tool_check_prompt.format(task_description=task_description, tool_description=tool_description)
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
    tool_matches = re.findall(r"<tools>(.+?)</tools>", all_content, re.DOTALL)
    if tool_matches:
        last_match = tool_matches[-1]
        tools = last_match.strip()
    else:
        tools = None

    return tools

if __name__ == "__main__":
    import yaml
    import json
    from langchain_core.runnables import RunnableConfig
    from configuration import ModelConfiguration

    tools = [
        {
            "name": "music_similarity_analyzer",
            "description": "Analyzes musical characteristics of an artist and finds similar artists based on audio features, genre, and style patterns",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist_name": {
                        "description": "Name of the artist to analyze for similarity matching",
                        "type": "string"
                    },
                    "music_features": {
                        "description": "Comma-separated list of musical features to prioritize in similarity analysis",
                        "type": "string"
                    }
                },
                "required": ["artist_name", "music_features"]
            },
            "outputs": {
                "type": "object",
                "properties": {
                    "similar_artists": {
                        "description": "List of similar artists with similarity scores and key matching features",
                        "type": "array"
                    },
                    "primary_genres": {
                        "description": "Main musical genres identified for the artist",
                        "type": "array"
                    },
                    "audio_profile": {
                        "description": "Analysis of tempo, energy, instrumentation and mood characteristics",
                        "type": "object"
                    }
                }
            }
        },
        {
            "name": "afrikaans_music_database",
            "description": "Searches comprehensive database of Afrikaans artists, albums, and songs with filtering capabilities",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "description": "Artist name, song title, or genre to search for",
                        "type": "string"
                    },
                    "filter_by": {
                        "description": "Filter criteria such as genre, era, or popularity",
                        "type": "string"
                    },
                    "result_limit": {
                        "description": "Maximum number of results to return",
                        "type": "integer"
                    }
                },
                "required": ["search_query"]
            },
            "outputs": {
                "type": "object",
                "properties": {
                    "search_results": {
                        "description": "Array of matching artists/songs with metadata and streaming links",
                        "type": "array"
                    },
                    "total_matches": {
                        "description": "Total number of matches found in database",
                        "type": "integer"
                    }
                }
            }
        },
        {
            "name": "content_generator",
            "description": "Creates well-structured explanations and summaries based on input data and template",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_data": {
                        "description": "Structured data to be transformed into narrative content",
                        "type": "object"
                    },
                    "content_template": {
                        "description": "Type of content to generate (playlist_explanation, artist_comparison, social_summary)",
                        "type": "string"
                    },
                    "tone_style": {
                        "description": "Desired writing style (enthusiastic, analytical, conversational)",
                        "type": "string"
                    }
                },
                "required": ["input_data", "content_template"]
            },
            "outputs": {
                "type": "object",
                "properties": {
                    "generated_content": {
                        "description": "The complete generated text content",
                        "type": "string"
                    },
                    "key_points": {
                        "description": "Extracted main points from the generated content",
                        "type": "array"
                    }
                }
            }
        }
    ]
    task_info = 'Create a "Virtual Band Discovery Engine" for the virtual alternative rock band Echo Mirage from Virtual City Harmonia. The system should analyze Echo Mirage\'s distinctive musical signature (characterized by atmospheric guitar work, introspective lyrics, and dynamic tempo shifts), identify 3-5 virtual artists with complementary styles, build a curated discovery playlist with detailed explanations of the musical connections, and generate a shareable summary for distribution on Virtual Music Platform Waveform.\n\nThe discovery should highlight specific musical elements like vocal textures, instrumental arrangements, and thematic depth that connect these virtual artists, providing fans with a meaningful pathway to explore similar music within the virtual alternative scene.'

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

    with open("configs/eval.yaml", 'r', encoding='utf-8') as f:
        agent_config = yaml.safe_load(f)
    
    step_config = create_step_config(agent_config, "FuzzyTaskAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)

    tool_check(cfg, json.dumps(tools), task_info)

