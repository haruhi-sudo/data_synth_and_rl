import argparse
import json
import os
import re

from datasets import Dataset

def make_map_fn(split):
    def process_fn(example, idx):
        index_id = example["id"]
        prompt = example["prompt"]
        question = prompt[1]["content"]
        task_background = example["task_background"]
        rubrics = example["rubrics"]

        prompt[1]["content"] = f"""{question} 
PAY Attention: Users cannot process too much information at once, don't ask for too much in a single interaction. 
/no_think"""
        
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
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "mock_tool": {
                        "create_kwargs": {
                            "tool_call_history_path": tool_call_history_path,
                            "index_id": index_id,
                            "tool_description": tool_description,
                            "task_background": task_background
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
    parser.add_argument("--local_dir", default="my_data/virtual_tool_use")
    parser.add_argument("--train_data_source", 
                        default="my_data/raw/tool_use/tool_use_train_verl.json")
    parser.add_argument("--test_data_source",
                        default="my_data/raw/tool_use/tool_use_val_verl.json")
    parser.add_argument("--tool_call_history_path", 
                        default="tmp/solve_tool_use")
    
    args = parser.parse_args()

    train_data_source = args.train_data_source
    test_data_source = args.test_data_source
    tool_call_history_path = args.tool_call_history_path
    
    with open(train_data_source, "r") as f:
        train_data = json.load(f)
    train_dataset = Dataset.from_list(train_data)

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    # Process train dataset
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    # Process test datasets
    with open(test_data_source, "r") as f:
        test_data = json.load(f)
    test_dataset = Dataset.from_list(test_data)
    
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    test_dataset.to_parquet(os.path.join(local_dir, "val.parquet"))
