import argparse
import json
import os

from datasets import Dataset


def make_map_fn(split):
    def process_fn(example, idx):
        prompt = example["prompt"]
        question = prompt[1]["content"]
        prompt[1]["content"] = f"""Question: {question}

Multiple tool invocations are permitted. Prior to each tool invocation, you must articulate your analytical reasoning.

Critical reminder: Tools operate in a stateless manner and will not preserve any variables, functions, or definitions from prior executions.

Reasoning step by step, write ONLY the final answer enclosed in <answer> and </answer> tags. Do not include any extra text outside the tags after the final answer. /no_think
"""
        answer = str(example["gt_answer"])
        
        data = {
            "data_source": "tool_agent",
            "agent_name": "tool_agent",
            "prompt": prompt,
            "ability": None,
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer,
                "question": question,
                "need_tools_kwargs": False,
                "interaction_kwargs": {
                    "query": question,
                    "ground_truth": answer,
                },
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="my_data/math-hard")
    parser.add_argument("--train_data_source", 
                        default="my_data/raw/reasoning/math_train_verl.json")
    parser.add_argument("--test_data_source",
                        default="my_data/raw/reasoning/aime24_val_verl.json")
    
    args = parser.parse_args()

    train_data_source = args.train_data_source
    test_data_source = args.test_data_source
    
    with open(train_data_source, "r") as f:
        train_data = json.load(f)
    train_dataset = Dataset.from_list(train_data)

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    # Process train dataset
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    # Process test dataset
    with open(test_data_source, "r") as f:
        test_data = json.load(f)
    test_dataset = Dataset.from_list(test_data)
    
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    test_dataset.to_parquet(os.path.join(local_dir, "val.parquet"))
