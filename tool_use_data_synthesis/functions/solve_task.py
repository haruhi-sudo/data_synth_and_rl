import json
import re
import copy
from openai import OpenAI
from typing import List, Dict, Optional

def call_llm_messages(
    messages: List[Dict[str, str]],
    api_base: Optional[str],
    api_key: Optional[str],
    model_name: str,
    max_tokens: int,
    temperature: float,
):
    client = OpenAI(api_key=api_key, base_url=api_base)
    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        extra_body={"max_completion_tokens": max_tokens}
    )
    response_content = response.choices[0].message.content
    messages.append(
        {"role": "assistant", "content": response_content}
    )

    return messages

def solve_task_by_tools(cfg, solve_history):
    solve_history = copy.deepcopy(solve_history)
    messages = call_llm_messages(
        messages=solve_history,
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        model_name=cfg.model_name,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
    
    think_and_tool_call = messages[-1]["content"]
    tool_call_matches = re.findall(r"<tool_call>(.+?)</tool_call>", think_and_tool_call, re.DOTALL)
    if tool_call_matches:
        last_match = tool_call_matches[-1]
        tool_call = last_match.strip()
    else:
        tool_call = None

    return think_and_tool_call, tool_call

if __name__ == "__main__":
    import yaml
    import json
    from langchain_core.runnables import RunnableConfig
    from configuration import ModelConfiguration

    task_info = 'Create a personalized digital fan tribute to the virtual Afrikaans band Windhowl by analyzing the emotional depth of one of their recent songs, designing a matching visual mood board, and sharing a curated post in the r/WindhowlFans community on Reddit—all within a 90-minute creative session.'
    tools_description = '[\n  {\n    "name": "LyriqScanAPI",\n    "description": "Retrieves lyrics and metadata for Afrikaans songs by artist and optional title. Simulates web scraping and music database lookup.",\n    "parameters": {\n      "type": "object",\n      "properties": {\n        "artist": {\n          "description": "Name of the musical artist",\n          "type": "string"\n        },\n        "song_title": {\n          "description": "Optional specific song title. If omitted, returns a recent popular song.",\n          "type": "string"\n        }\n      },\n      "required": ["artist"]\n    },\n    "outputs": {\n      "type": "object",\n      "properties": {\n        "title": {\n          "description": "Song title",\n          "type": "string"\n        },\n        "lyrics": {\n          "description": "Full lyrics in Afrikaans",\n          "type": "string"\n        }\n      },\n      "required": ["title", "lyrics"]\n    }\n  },\n  {\n    "name": "MoodBoardGenerator",\n    "description": "Generates a virtual mood board based on theme and emotional tone. Returns a simulated ID and textual description of visual elements.",\n    "parameters": {\n      "type": "object",\n      "properties": {\n        "primary_theme": {\n          "description": "Main theme to visualize (e.g., \'Heimwee\', \'Liefde\')",\n          "type": "string"\n        },\n        "emotional_tone": {\n          "description": "Emotion to reflect in visuals (e.g., \'melancholies\', \'hoopvol\')",\n          "type": "string"\n        },\n        "color_preference": {\n          "description": "Optional color scheme (e.g., \'warm\', \'monochrome\')",\n          "type": "string"\n        }\n      },\n      "required": ["primary_theme", "emotional_tone"]\n    },\n    "outputs": {\n      "type": "object",\n      "properties": {\n        "moodboard_id": {\n          "description": "Unique identifier for the generated mood board",\n          "type": "string"\n        },\n        "description": {\n          "description": "Textual description of the mood board\'s visual elements and style",\n          "type": "string"\n        }\n      },\n      "required": ["moodboard_id", "description"]\n    }\n  },\n  {\n    "name": "SocialPostCraft",\n    "description": "Publishes a post to a fan community platform with image, caption, and hashtags. Simulates social media posting.",\n    "parameters": {\n      "type": "object",\n      "properties": {\n        "platform": {\n          "description": "Target platform (e.g., \'Reddit\', \'Facebook\')",\n          "type": "string"\n        },\n        "community": {\n          "description": "Name of the group or subreddit",\n          "type": "string"\n        },\n        "caption": {\n          "description": "Text caption for the post",\n          "type": "string"\n        },\n        "image_url": {\n          "description": "URL of image to attach (e.g., from MoodBoardGenerator)",\n          "type": "string"\n        },\n        "hashtags": {\n          "description": "List of relevant hashtags",\n          "type": "array",\n          "items": {\n            "type": "string"\n          }\n        }\n      },\n      "required": ["platform", "community", "caption", "image_url"]\n    },\n    "outputs": {\n      "type": "object",\n      "properties": {\n        "status": {\n          "description": "Status of the post (e.g., \'published\')",\n          "type": "string"\n        },\n        "post_url": {\n          "description": "URL to the published post",\n          "type": "string"\n        }\n      },\n      "required": ["status", "post_url"]\n    }\n  },\n  {\n    "name": "FanSentimentTracker",\n    "description": "Simulates retrieval of current sentiment trends about an artist in fan communities using social listening tools.",\n    "parameters": {\n      "type": "object",\n      "properties": {\n        "artist": {\n          "description": "Name of the musical artist",\n          "type": "string"\n        }\n      },\n      "required": ["artist"]\n    },\n    "outputs": {\n      "type": "object",\n      "properties": {\n        "overall_sentiment": {\n          "description": "Aggregated sentiment (e.g., \'positive\', \'mixed\')",\n          "type": "string"\n        },\n        "trend_summary": {\n          "description": "Brief textual summary of recent discussions",\n          "type": "string"\n        }\n      },\n      "required": ["overall_sentiment", "trend_summary"]\n    }\n  },\n  {\n    "name": "AudioSnipGenerator",\n    "description": "Generates a short audio clip from a song. Returns a virtual ID and status. For simulation only.",\n    "parameters": {\n      "type": "object",\n      "properties": {\n        "artist": {\n          "description": "Name of the artist",\n          "type": "string"\n        },\n        "song_title": {\n          "description": "Title of the song",\n          "type": "string"\n        },\n        "duration_seconds": {\n          "description": "Length of clip in seconds (max 30)",\n          "type": "integer"\n        }\n      },\n      "required": ["artist", "song_title"]\n    },\n    "outputs": {\n      "type": "object",\n      "properties": {\n        "status": {\n          "description": "Generation status (e.g., \'success\')",\n          "type": "string"\n        },\n        "clip_id": {\n          "description": "Unique identifier for the generated audio clip",\n          "type": "string"\n        }\n      },\n      "required": ["status", "clip_id"]\n    }\n  }\n]'
    tools_format = json.loads(tools_description)
    tools_description = ""

    for tool in tools_format:
        tool.pop("outputs")
        tools_description += json.dumps({"type": "function", "function": tool}) + "\n"

    _STEP_CONFIG_KEYS = ("model_name", "temperature", "max_tokens")

    def create_step_config(base_config, step_name):
        step_model = base_config["step_models"][step_name]
        configurable = {k: v for k, v in step_model.items() if k in _STEP_CONFIG_KEYS}
        configurable["api_configs"] = base_config["api_configs"]
        return {"configurable": configurable}

    with open("configs/eval.yaml", 'r', encoding='utf-8') as f:
        agent_config = yaml.safe_load(f)
    
    step_config = create_step_config(agent_config, "FuzzyTaskAgent")
    cfg = ModelConfiguration.from_runnable_config(step_config)


    system_prompt = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{available_tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""
    system_prompt = system_prompt.format(available_tools=tools_description)

    prompt = f"Task Description: {task_info}. Please use only one tool at a time, and provide your step-by-step reasoning before using any tool."

    solve_history = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    solve_task_by_tools(cfg, solve_history)
