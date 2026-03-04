import json
import re

from my_script.data_process.base import convert_dataset, make_base_parser


def normalize_examples_before_arrow(examples):
    """Serialize tool_return_expected to JSON string to avoid Arrow type conflicts.

    HuggingFace Datasets / PyArrow requires each column to have a consistent type.
    In raw json, `tool_return_expected[*].input` can be either a dict (struct)
    or a string, which makes Arrow fail.
    """
    for ex in examples:
        tre = ex.get("tool_return_expected", None)
        if tre is not None and not isinstance(tre, str):
            ex["tool_return_expected"] = json.dumps(tre, indent=4, ensure_ascii=False)
    return examples


def make_map_fn(split):
    def process_fn(example, idx):
        index_id = example["id"]
        prompt = example["prompt"]
        question = prompt[1]["content"]
        system_prompt = prompt[0]["content"]

        policy_matches = re.findall(r"<policy>(.*?)</policy>", system_prompt, re.DOTALL)
        policy = policy_matches[-1].strip()

        task_background = example["task_background"]
        rubrics = example["rubrics"]
        test_policy = example["test_policy"]
        user_escape_strategy = example["user_escape_strategy"]
        tool_return_expected = example["tool_return_expected"]
        if not isinstance(tool_return_expected, str):
            tool_return_expected = json.dumps(tool_return_expected, indent=4, ensure_ascii=False)

        prompt[0]["content"] = f"""{system_prompt}

If the user end the conversation with "###STOP", please output your final answer wrapped in <answer> and </answer> tags.
If you select "###TRANSFER_TO_HUMAN", please also output your conclusion wrapped in <answer> and </answer> tags.
The content wrapped in <answer> and </answer> tags should be concise(one or two sentences).

CRITICAL:

You cannot terminate the task yourself. Only output <answer></answer> tag when the user outputs ###STOP or you decide to ###TRANSFER_TO_HUMAN. This rule is essential, self-terminating will cause complete task failure.

When you need to ask the user for more information, you should wrap the question in <question> and </question> tags.

/no_think
"""
        prompt[1]["content"] += " When you need to ask the user for more information, you should wrap the question in <question> and </question> tags."
        tool_description = prompt[0]["content"]
        tools_matches = re.findall(r"<tools>(.*?)</tools>", tool_description, re.DOTALL)
        tool_description = tools_matches[-1].strip()

        return {
            "data_source": "my_tool_agent_with_user",
            "agent_name": "my_tool_agent_with_user",
            "prompt": prompt,
            "ability": None,
            "reward_model": {
                "style": "rule",
                "ground_truth": "None",
            },
            "extra_info": {
                "split": split,
                "index": index_id,
                "question": question,
                "rubrics": rubrics,
                "policy": policy,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "mock_tool": {
                        "create_kwargs": {
                            "tool_description": tool_description,
                            "task_background": task_background,
                            "test_policy": test_policy,
                            "user_escape_strategy": user_escape_strategy,
                            "tool_return_expected": tool_return_expected,
                        },
                    },
                },
                "interaction_kwargs": {
                    "query": question,
                },
            },
        }

    return process_fn


if __name__ == "__main__":
    parser = make_base_parser(
        default_output_dir="my_data/tmp/tool_use_data_filtered",
        default_train_source="my_data/tmp/tool_use_data_filtered.json",
        default_test_source="my_data/tmp/tool_use_data_filtered_val.json",
    )
    args = parser.parse_args()

    convert_dataset(
        train_source=args.train_data_source,
        test_source=args.test_data_source,
        output_dir=args.local_dir,
        make_map_fn=make_map_fn,
        preprocess_fn=normalize_examples_before_arrow,
    )
