import argparse
import json
import os
import re

from datasets import Dataset

def normalize_examples_before_arrow(examples):
    """
    HuggingFace Datasets / PyArrow requires each column to have a consistent type.
    In our raw json, `tool_return_expected[*].input` can be either a dict (struct)
    or a string (non-struct), which makes Arrow fail with:
      ArrowInvalid: cannot mix struct and non-struct, non-null values

    Easiest robust fix: serialize the whole `tool_return_expected` field to a JSON
    string BEFORE calling Dataset.from_list().
    """
    changed = 0
    for ex in examples:
        tre = ex.get("tool_return_expected", None)
        if tre is not None and not isinstance(tre, str):
            ex["tool_return_expected"] = json.dumps(tre, indent=4, ensure_ascii=False)
            changed += 1
    return examples, changed

def make_map_fn(split):
    def process_fn(example, idx):
        index_id = example["id"]
        prompt = example["prompt"]
        question = prompt[1]["content"]
        system_prompt = prompt[0]["content"]

        # extrac <policy> and </policy> from system_prompt
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
        
        data = {
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
                            "tool_call_history_path": tool_call_history_path,
                            "index_id": index_id,
                            "tool_description": tool_description,
                            "task_background": task_background,
                            "test_policy": test_policy,
                            "user_escape_strategy": user_escape_strategy,
                            "tool_return_expected": tool_return_expected
                        },
                    },
                },
                "interaction_kwargs": {
                    "query": question,
                },
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="my_data/tmp/policy_tool_use_v3.2_easy")
    parser.add_argument("--train_data_source", 
                        default="my_data/tmp/raw/policy_tool_use_data_v3.2_easy_verl.json")
    parser.add_argument("--test_data_source",
                        default="my_data/tmp/raw/policy_tool_use_data_v3.2_easy_verl_val.json")
    parser.add_argument("--tool_call_history_path", 
                        default="tmp/solve_tool_use")
    
    args = parser.parse_args()

    train_data_source = args.train_data_source
    test_data_source = args.test_data_source
    tool_call_history_path = args.tool_call_history_path
    
    with open(train_data_source, "r") as f:
        train_data = json.load(f)
    train_data, _changed = normalize_examples_before_arrow(train_data)
    train_dataset = Dataset.from_list(train_data)

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    # Process train dataset
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    # Process test datasets
    with open(test_data_source, "r") as f:
        test_data = json.load(f)
    test_data, _changed = normalize_examples_before_arrow(test_data)
    test_dataset = Dataset.from_list(test_data)
    
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    test_dataset.to_parquet(os.path.join(local_dir, "val.parquet"))
