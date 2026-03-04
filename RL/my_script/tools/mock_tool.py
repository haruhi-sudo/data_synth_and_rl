import json
import logging
import os
import re
import threading
import time
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray
from dotenv import load_dotenv

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

from my_script.prompts.mock_user import MOCK_USER_PROMPT
from my_script.prompts.tool_simulation import TOOL_SIMULATION_PROMPT
from my_script.utils.llm_client import call_llm_sync
from my_script.utils.message_parser import extract_tool_history, extract_user_conversation

load_dotenv()
load_dotenv(".local.env", override=True)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


def mock_user_response(task_background, test_policy, user_escape_strategy, history_messages, user_timeout: float = 300.0):
    """Generate mock user response with timeout control."""
    conversation_history = extract_user_conversation(history_messages)

    user_prompt = MOCK_USER_PROMPT.format(
        task_background=task_background,
        test_policy=test_policy,
        user_escape_strategy=user_escape_strategy,
        conversation_history=conversation_history,
    )

    response_content = call_llm_sync(
        user_prompt=user_prompt,
        system_prompt="",
        api_base=os.environ.get("MOCK_USER_API_BASE", ""),
        api_key=os.environ.get("MOCK_USER_API_KEY", ""),
        model_name=os.environ.get("MOCK_USER_MODEL_NAME", ""),
        max_tokens=2048,
        timeout=user_timeout,
    )

    user_response_matches = re.findall(r"<reply>(.+?)</reply>", response_content, re.DOTALL)
    if user_response_matches:
        return user_response_matches[-1].strip()
    else:
        logger.warning("Failed to parse user response, use the raw content instead")
        logger.warning(f"User response: {response_content}")
        return response_content


class PoolMode(Enum):
    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
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
    def __init__(self, enable_global_rate_limit=True, rate_limit=10, acquire_timeout=30):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None
        self.acquire_timeout = acquire_timeout

    def _init_rate_limit(self, rate_limit):
        return TokenBucketWorker.options(name="mock-tool-rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        acquire_start_time = time.time()
        try:
            acquire_ref = self.rate_limit_worker.acquire.remote()
            ready, not_ready = ray.wait([acquire_ref], timeout=self.acquire_timeout, num_returns=1)
            if not ready:
                acquire_elapsed = time.time() - acquire_start_time
                logger.error(f"[TIMEOUT_DIAGNOSIS] Rate limit acquire timeout: waited {acquire_elapsed:.2f}s (limit: {self.acquire_timeout}s)")
                return ToolResponse(text="Timeout, please try again later")
            ray.get(acquire_ref)
        except TimeoutError:
            return ToolResponse(text="Timeout, please try again later")
        except Exception as e:
            acquire_elapsed = time.time() - acquire_start_time
            logger.error(f"[TIMEOUT_DIAGNOSIS] Rate limit acquire failed after {acquire_elapsed:.2f}s: {e}")
            return ToolResponse(text="Timeout, please try again later")

        with ExitStack() as stack:
            stack.callback(self.rate_limit_worker.release.remote)
            fn_start_time = time.time()
            try:
                return fn(*fn_args, **fn_kwargs)
            except Exception as e:
                fn_elapsed = time.time() - fn_start_time
                logger.warning(f"[TIMEOUT_DIAGNOSIS] Function execution failed after {fn_elapsed:.2f}s: {e}")
                return ToolResponse(text="Timeout, please try again later")


def init_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, acquire_timeout=30, mode: PoolMode = PoolMode.ThreadMode
):
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(ExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit, acquire_timeout=acquire_timeout)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class MockSandboxFusionTool(BaseTool):
    """A mock tool that uses an LLM to simulate virtual tool calls and user interactions."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

        self._instance_dict = {}
        self.num_workers = config.get("num_workers", 10)
        self.rate_limit = config.get("rate_limit", 10)
        self.default_timeout = config.get("default_timeout", 30)
        self.acquire_timeout = config.get("acquire_timeout", 30)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            acquire_timeout=self.acquire_timeout,
            mode=PoolMode.ThreadMode,
        )
        logger.info(f"Init SandboxFusionTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        create_kwargs = kwargs.get("create_kwargs", {})
        tool_description = create_kwargs.get("tool_description", "")
        task_background = create_kwargs.get("task_background", "")
        test_policy = create_kwargs.get("test_policy", "")
        user_escape_strategy = create_kwargs.get("user_escape_strategy", "")
        tool_return_expected = create_kwargs.get("tool_return_expected", "")

        if "clarification case" in test_policy:
            test_policy = ""

        if instance_id is None:
            instance_id = str(uuid4())

        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": [],
            "tool_description": tool_description,
            "task_background": task_background,
            "test_policy": test_policy,
            "user_escape_strategy": user_escape_strategy,
            "tool_return_expected": tool_return_expected,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        mock_tool_name_and_args = parameters.get("mock_tool_name_and_args", "")
        total_timeout = parameters.get("timeout", self.default_timeout)
        history_messages = parameters.get("history_messages", "")
        if not isinstance(mock_tool_name_and_args, str):
            mock_tool_name_and_args = str(mock_tool_name_and_args)

        result = await self.execution_pool.execute.remote(self.execute_mock_tool, instance_id, mock_tool_name_and_args, history_messages, total_timeout)
        return result, None, None

    def _simulate_tool_response(self, instance_id: str, mock_tool_name_and_args: dict,
                                history: str, timeout: float) -> str:
        """Call the LLM to simulate the response of a virtual tool."""
        tool_description = self._instance_dict[instance_id]["tool_description"]
        tool_return_expected = self._instance_dict[instance_id]["tool_return_expected"]

        user_prompt = TOOL_SIMULATION_PROMPT.format(
            query=mock_tool_name_and_args,
            tools=tool_description,
            history=history,
            world_state=tool_return_expected,
        )
        response_content = call_llm_sync(
            user_prompt=user_prompt,
            system_prompt="",
            api_base=os.environ.get("MOCK_TOOL_API_BASE", ""),
            api_key=os.environ.get("MOCK_TOOL_API_KEY", ""),
            model_name=os.environ.get("MOCK_TOOL_MODEL_NAME", ""),
            max_tokens=2048,
            timeout=timeout,
        )

        tool_response_matches = re.findall(
            r"<simulated_tool_response>(.+?)</simulated_tool_response>", response_content, re.DOTALL
        )
        if tool_response_matches:
            return tool_response_matches[-1].strip()

        parts = response_content.strip().split("<simulated_tool_response>")
        if len(parts) == 1:
            logger.warning("Failed to parse tool response")
            return "Failed to call the tool, please try again."
        logger.warning(f"Tool response: {parts[-1].strip()}")
        return parts[-1].strip()

    def execute_mock_tool(self, instance_id, mock_tool_name_and_args, history_messages, timeout=30):
        """Execute mock tool with timeout control."""
        mock_tool_name_and_args = json.loads(mock_tool_name_and_args)

        # Act as a user
        if mock_tool_name_and_args["name"] == "mock_user":
            task_background = self._instance_dict[instance_id]["task_background"]
            test_policy = self._instance_dict[instance_id]["test_policy"]
            user_escape_strategy = self._instance_dict[instance_id]["user_escape_strategy"]
            user_response = mock_user_response(
                task_background, test_policy, user_escape_strategy, history_messages, user_timeout=timeout
            )
            return ToolResponse(text=user_response) if user_response else ToolResponse(text="None")

        # Act as a tool
        tool_history = extract_tool_history(history_messages)
        tool_response = self._simulate_tool_response(
            instance_id, mock_tool_name_and_args, tool_history, timeout
        )
        return ToolResponse(text=tool_response) if tool_response else ToolResponse(text="None")

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
