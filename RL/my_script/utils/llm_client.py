"""Unified LLM API client with sync and async interfaces."""

import asyncio
import json
import logging
import os
import time
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from openai import APITimeoutError as OpenAIAPITimeoutError

load_dotenv()
load_dotenv(".local.env", override=True)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Async client defaults
ASYNC_MAX_CONCURRENT = 16
ASYNC_MAX_RETRIES = 3
ASYNC_BASE_DELAY = 10
ASYNC_TIMEOUT = 150.0

_async_semaphore = asyncio.Semaphore(ASYNC_MAX_CONCURRENT)
_async_http_client: Optional[httpx.AsyncClient] = None


def _safe_preview(text: str, limit: int = 1200) -> str:
    if text is None:
        return "None"
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


# ---------------------------------------------------------------------------
# Sync client (used by mock_tool.py via OpenAI SDK)
# ---------------------------------------------------------------------------

def call_llm_sync(
    user_prompt: str,
    system_prompt: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: str = "",
    max_tokens: int = 2048,
    temperature: float = 0.9,
    timeout: float = 300.0,
) -> str:
    """Call LLM API synchronously via OpenAI SDK. Returns the assistant's response content."""
    start_time = time.time()
    try:
        client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout, max_retries=0)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except (TimeoutError, httpx.TimeoutException, OpenAIAPITimeoutError) as e:
        elapsed = time.time() - start_time
        logger.error(f"[TIMEOUT] API call timed out after {elapsed:.2f}s (limit: {timeout}s): {type(e).__name__}: {e}")
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[ERROR] API call failed after {elapsed:.2f}s: {type(e).__name__}: {e}")
        raise


# ---------------------------------------------------------------------------
# Async client (used by reward_function.py via httpx)
# ---------------------------------------------------------------------------

async def _get_async_client() -> httpx.AsyncClient:
    global _async_http_client
    if _async_http_client is None or _async_http_client.is_closed:
        _async_http_client = httpx.AsyncClient(timeout=httpx.Timeout(ASYNC_TIMEOUT))
    return _async_http_client


async def close_async_client():
    global _async_http_client
    if _async_http_client is not None and not _async_http_client.is_closed:
        await _async_http_client.aclose()
    _async_http_client = None


async def call_llm_async(
    user_prompt: str,
    system_prompt: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: str = "",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    response_format: Optional[dict[str, Any]] = None,
) -> str:
    """Call LLM API asynchronously via httpx. Returns the assistant's response content."""
    async with _async_semaphore:
        if not api_base:
            raise ValueError("api_base is required")
        if not api_key:
            raise ValueError("api_key is required")

        url = f"{api_base.rstrip('/')}/chat/completions"
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        client = await _get_async_client()
        last_exc: Optional[Exception] = None

        for attempt in range(1, ASYNC_MAX_RETRIES + 1):
            try:
                resp = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )

                if resp.status_code >= 400:
                    logger.warning(
                        "LLM API non-2xx (attempt %d/%d). status=%s url=%s model=%s body=%s",
                        attempt, ASYNC_MAX_RETRIES, resp.status_code, url, model_name,
                        _safe_preview(resp.text),
                    )

                resp.raise_for_status()

                try:
                    data = resp.json()
                except Exception as je:
                    logger.error(
                        "LLM API invalid JSON (attempt %d/%d). status=%s url=%s model=%s raw=%s err=%r",
                        attempt, ASYNC_MAX_RETRIES, resp.status_code, url, model_name,
                        _safe_preview(resp.text), je,
                    )
                    raise

                try:
                    content = data["choices"][0]["message"]["content"]
                except Exception as pe:
                    logger.error(
                        "LLM API unexpected schema (attempt %d/%d). url=%s model=%s data_preview=%s err=%r",
                        attempt, ASYNC_MAX_RETRIES, url, model_name,
                        _safe_preview(json.dumps(data, ensure_ascii=False)), pe,
                    )
                    raise

                if content is None:
                    raise ValueError("API returned None as content")

                return content

            except Exception as e:
                last_exc = e
                if attempt < ASYNC_MAX_RETRIES:
                    delay = ASYNC_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "call_llm_async failed (attempt %d/%d). url=%s model=%s err=%r; retrying in %.2fs",
                        attempt, ASYNC_MAX_RETRIES, url, model_name, e, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Failed after %d attempts. url=%s model=%s last_err=%r",
                        ASYNC_MAX_RETRIES, url, model_name, e,
                    )

        return "No message"
