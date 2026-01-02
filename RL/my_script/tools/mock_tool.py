import json
import logging
import os
import re
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray
from dotenv import load_dotenv
from openai import OpenAI
from filelock import FileLock

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.reward_score.sandbox_fusion.utils import _process_single_case
from verl.utils.rollout_trace import rollout_trace_op

load_dotenv()
load_dotenv(".local.env", override=True)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")

def call_llm_api(
    user_prompt: str,
    system_prompt: str,
    api_base: Optional[str],
    api_key: Optional[str],
    model_name: str,
    max_tokens: int,
    temperature: float=0.9,
):
    client = OpenAI(api_key=api_key, base_url=api_base)
    dynamic_model_id = model_name
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = client.chat.completions.create(
        model=dynamic_model_id,
        messages=messages,
        temperature=temperature,
        extra_body={"max_completion_tokens": max_tokens},
    )
    response_content = response.choices[0].message.content
    messages.append(
        {"role": "assistant", "content": response_content}
    )

    return messages

def mock_user_response(task_background, history_messages):
    user_prompt = f"""
You are an agent user who wants to accomplish a task with tools.

Here is the background information you know:
{task_background}

Your interaction history with the agent so far is:
{history_messages}


### Requirements:

1. Please use the background information to reply to the agent's question **in one sentence**. Make sure to reveal the information gradually. Share only a small part of the background at a time rather than giving everything away at once.
2. Don't give the agent any hint about the solution or the tools used.

Write your reply in the following format:
<reply>
(Your reply)
</reply>
"""
    messages = call_llm_api(
        user_prompt=user_prompt,
        system_prompt="",
        api_base=os.environ.get("MOCK_USER_API_BASE", ""),
        api_key=os.environ.get("MOCK_USER_API_KEY", ""),
        model_name="",
        max_tokens=10240
    )

    all_content = messages[-1]["content"]
    user_response_matches = re.findall(r"<reply>(.+?)</reply>", all_content, re.DOTALL)
    if user_response_matches:
        last_match = user_response_matches[-1]
        user_response = last_match.strip()
    else:
        # breakpoint()
        user_response = None

    return user_response


class PoolMode(Enum):
    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        # this only used for observalability
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        return self.current_count


class ExecutionWorker:
    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        # TODO validation for rate_limit
        # A Singleton Rate Limitor
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        with ExitStack() as stack:
            stack.callback(self.rate_limit_worker.release.remote)
            ray.get(self.rate_limit_worker.acquire.remote())
            try:
                return fn(*fn_args, **fn_kwargs)
            except Exception as e:
                # TODO we should make this available to the tool caller
                logger.warning(f"Error when executing code: {e}")


def init_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(ExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")
        # return ray.util.multiprocessing.Pool(processes=num_workers)


class MockSandboxFusionTool(BaseTool):
    """A tool for executing the code using sanbox fusion image.

    - `get_openai_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "mock_tool",
                "description": "An LLM imitates a tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The query",
                        },
                    },
                    "required": ["query"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)

        self._instance_dict = {}
        # TODO: better documentation for the config
        self.num_workers = config.get("num_workers", 10)
        self.rate_limit = config.get("rate_limit", 10)
        self.default_timeout = config.get("default_timeout", 30)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        self.sandbox_fusion_url = config.get("sandbox_fusion_url", "") or os.environ.get("MOCK_TOOL_API_BASE", "")
        self.sandbox_fusion_key = config.get("sandbox_fusion_key", "") or os.environ.get("MOCK_TOOL_API_KEY", "")

        if self.sandbox_fusion_url == "":
            raise ValueError("sandbox_fusion_url is not set")
        log_msg = f"Init SandboxFusionTool with config: {config}"
        logger.info(log_msg)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        create_kwargs = kwargs.get("create_kwargs", {})
        tool_call_history_path_root = create_kwargs.get("tool_call_history_path", "")
        tool_description = create_kwargs.get("tool_description", "")
        index_id = create_kwargs.get("index_id", "")
        task_background = create_kwargs.get("task_background", "")

        tool_call_history_path = os.path.join(tool_call_history_path_root, index_id, "tool_call_history.json")

        if os.path.exists(tool_call_history_path):
            with open(tool_call_history_path) as f:
                tool_call_history = json.load(f)
        else:
            os.makedirs(os.path.dirname(tool_call_history_path), exist_ok=True)
            tool_call_history = []
            with open(tool_call_history_path, "w") as f:
                json.dump(tool_call_history, f)


        if instance_id is None:
            instance_id = str(uuid4())

        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": [],
            "tool_call_history_path": tool_call_history_path,
            "tool_description": tool_description,
            "task_background": task_background,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        mock_tool_name_and_args = parameters.get("mock_tool_name_and_args", "")
        timeout = parameters.get("timeout", self.default_timeout)
        history_messages = parameters.get("history_messages", [])
        if not isinstance(mock_tool_name_and_args, str):
            mock_tool_name_and_args = str(mock_tool_name_and_args)

        result = await self.execution_pool.execute.remote(self.execute_mock_tool, instance_id, mock_tool_name_and_args, history_messages, timeout)

        if result is None or result.text == "None":
            logger.warning("Response is None for mock tools or users")
        # sandbox has no score or metrics, use Nones
        return result, None, None

    def execute_mock_tool(self, instance_id, mock_tool_name_and_args, history_messages, timeout=30):
        def mock_tool_response(mock_tool_name_and_args, tool_description, save_tool_call_path):
            # breakpoint()
            lock_path = save_tool_call_path + ".lock"
            file_lock = FileLock(lock_path, timeout=10)

            with file_lock:
                with open(save_tool_call_path, "r") as f:
                    world_state = json.load(f)
            
            tool_simulation_prompt_with_memory = """
You are the prophet of the virtual world, knowing all affairs and details of the virtual world, and able to remember all previously established background and states of the virtual world. Now, an external intelligent agent will request information from you via a tool call. You need to simulate the tool based on the tool call information and strictly return output that matches the required format.

### Virtual tools you need to simulate (including function descriptions)
{tools}

### World state (past memory)
The following is the known and established information of the virtual world so far. This information was generated by past tool calls or events, and must be strictly followed. You must not deny it or generate conflicts with it:
{world_state}

### Current tool call request from the intelligent agent
{query}

### Response rules
1. Accurately simulate the tool's return  
- Generate the response strictly according to the tool's function description and parameters  
- The output must be consistent with past tool calls. If multiple tool calls in the world state match the current query, you should randomly select one to reuse
- The format must comply with the return parameter structure of the tool

2. Do not add extra explanations or non-tool output  
- Do not explain your actions or reasoning process
- Do not output narrative paragraphs unless the tool itself is designed to return narrative information
- Only return what the tool would return

3. Context length considerations  
- Because the model has limited context capacity and all past query information is included in the world state, tool responses should not be excessively long  
- At the same time, to ensure the intelligent agent receives sufficient information, the responses must not be overly brief

### Output Format
You MUST output exactly two tagged sections:

<is_new>
[true or false]
- false: if you found and reused a response from an existing similar tool call in world state
- true: if you generated a new response because no similar tool call was found
</is_new>

<response>
[The tool response here - if No prior tool call found, you MUST generate a new response; Else, reponse from the tool call]
</response>

Remember: Always check world state FIRST before generating anything new! If multiple tool calls in the world state match the current query (same tool name and similar/compatible parameters), you should **randomly select ONE** of them to reuse.
"""
            user_prompt = tool_simulation_prompt_with_memory.format(
                query=mock_tool_name_and_args, tools=tool_description,
                world_state=json.dumps(world_state)
            )
            messages = call_llm_api(
                user_prompt=user_prompt,
                system_prompt="",
                api_base=os.environ.get("MOCK_TOOL_API_BASE", ""),
                api_key=os.environ.get("MOCK_TOOL_API_KEY", ""),
                model_name="",
                max_tokens=20480
            )

            all_content = messages[-1]["content"]
            
            # Extract tool response
            tool_response_matches = re.findall(r"<response>(.+?)</response>", all_content, re.DOTALL)
            if tool_response_matches:
                last_match = tool_response_matches[-1]
                tool_response = last_match.strip()
            else:
                tool_response = None
            
            # Extract is_new flag
            is_new_matches = re.findall(r"<is_new>(.+?)</is_new>", all_content, re.DOTALL)
            is_new = False
            if is_new_matches:
                is_new_str = is_new_matches[-1].strip().lower()
                is_new = is_new_str == "true"

            if is_new:
                if len(world_state) < 100:
                    world_state.append(f"Query:\n{mock_tool_name_and_args}\nResponse:\n{tool_response}")
                    with file_lock:
                        with open(save_tool_call_path, "w") as f:
                            json.dump(world_state, f, indent=4, ensure_ascii=False)
            
            return tool_response
        
        mock_tool_name_and_args = json.loads(mock_tool_name_and_args)
        # Act as an user
        if mock_tool_name_and_args["name"] == "mock_user":
            task_background = self._instance_dict[instance_id]["task_background"]
            user_response = mock_user_response(task_background, history_messages)

            return ToolResponse(text=user_response) if user_response else ToolResponse(text="None")
        
        # Act as a tool
        # history_interactions = self._instance_dict[instance_id]["tool_call_history"]
        tool_description = self._instance_dict[instance_id]["tool_description"]

        tool_response = mock_tool_response(
            mock_tool_name_and_args, tool_description, 
            save_tool_call_path=self._instance_dict[instance_id]["tool_call_history_path"]
        )

        return ToolResponse(text=tool_response) if tool_response else ToolResponse(text="None")

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

