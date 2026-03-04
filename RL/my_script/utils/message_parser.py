"""Shared utilities for parsing chat messages in <|im_start|>role\ncontent<|im_end|> format."""

import re

def parse_chat_turns(raw_text: str) -> list[tuple[str, str]]:
    """Parse raw chat text into (role, content) pairs, stripping system messages.

    Handles the format:
        <|im_start|>role
        content
        <|im_end|>

    Returns list of (role, content) tuples for user and assistant turns only.
    """
    # Remove system messages
    cleaned = re.sub(
        r'<\|im_start\|>system\n.*?<\|im_end\|>\n?',
        '', raw_text, flags=re.DOTALL
    )
    return re.findall(
        r'<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>',
        cleaned, flags=re.DOTALL
    )


def extract_user_conversation(raw_text: str) -> str:
    """Extract real user messages and agent <question> content.

    Used by mock user simulation to understand conversation context.
    Skips tool response turns (user turns containing <tool_response>).
    """
    turns = parse_chat_turns(raw_text)
    parts = []
    turn_number = 0

    for role, content in turns:
        if role == 'user':
            if '<tool_response>' in content:
                continue
            turn_number += 1
            parts.append(f"[Round {turn_number} - You]:\n{content.strip()}")
        elif role == 'assistant':
            questions = re.findall(r'<question>(.*?)</question>', content, flags=re.DOTALL)
            if questions:
                question_text = '\n'.join(q.strip() for q in questions)
                parts.append(f"[Round {turn_number} - Agent]:\n{question_text}")

    return '\n\n'.join(parts)


def extract_tool_history(raw_text: str) -> str:
    """Extract tool calls and tool responses from chat history.

    Used by tool simulation to maintain state consistency.
    Tool responses are wrapped in <tool_response> tags inside user turns.
    """
    turns = parse_chat_turns(raw_text)
    parts = []
    turn_number = 0

    for role, content in turns:
        if role == 'assistant':
            tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', content, flags=re.DOTALL)
            for tc in tool_calls:
                turn_number += 1
                parts.append(f"[Tool Call {turn_number}]:\n{tc.strip()}")
        elif role == 'user':
            tool_responses = re.findall(r'<tool_response>(.*?)</tool_response>', content, flags=re.DOTALL)
            for tr in tool_responses:
                parts.append(f"[Tool Response {turn_number}]:\n{tr.strip()}")

    return '\n\n'.join(parts)


def extract_solution_summary(raw_text: str) -> str:
    """Extract all structured content from a solution for judge evaluation.

    Extracts: user messages, agent <question>/<tool_call>/<answer> content,
    tool responses, and termination markers (###STOP, ###TRANSFER_TO_HUMAN).
    Strips thinking blocks and other assistant reasoning text.
    """
    turns = parse_chat_turns(raw_text)
    parts = []
    turn_number = 0

    for role, content in turns:
        if role == 'user':
            tool_responses = re.findall(r'<tool_response>(.*?)</tool_response>', content, flags=re.DOTALL)
            if tool_responses:
                for tr in tool_responses:
                    parts.append(f"[Tool Response]:\n{tr.strip()}")
            else:
                turn_number += 1
                parts.append(f"[Round {turn_number} - User]:\n{content.strip()}")
        elif role == 'assistant':
            questions = re.findall(r'<question>(.*?)</question>', content, flags=re.DOTALL)
            tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', content, flags=re.DOTALL)
            answers = re.findall(r'<answer>(.*?)</answer>', content, flags=re.DOTALL)

            for q in questions:
                parts.append(f"[Round {turn_number} - Agent Question]:\n{q.strip()}")
            for tc in tool_calls:
                parts.append(f"[Tool Call - Agent]:\n{tc.strip()}")
            for a in answers:
                parts.append(f"[Agent Answer]:\n{a.strip()}")

            if "###TRANSFER_TO_HUMAN" in content:
                parts.append("[Agent Escalated to Human]: ###TRANSFER_TO_HUMAN")

    return '\n\n'.join(parts)
