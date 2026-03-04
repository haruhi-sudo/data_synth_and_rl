import argparse

from my_script.data_process.base import convert_dataset, make_base_parser


def make_map_fn(split):
    def process_fn(example, idx):
        prompt = example["prompt"]
        question = prompt[1]["content"]
        prompt[1]["content"] = f"""Question: {question}

Multiple tool invocations are permitted. Prior to each tool invocation, you must articulate your analytical reasoning.

Critical reminder: Tools operate in a stateless manner and will not preserve any variables, functions, or definitions from prior executions.

Reasoning step by step, keep your thinking process concise and minimal, and write ONLY the final answer enclosed in <answer> and </answer> tags. Do not include any extra text outside the tags after the final answer. /no_think
"""
        answer = str(example["gt_answer"])

        return {
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

    return process_fn


if __name__ == "__main__":
    parser = make_base_parser(
        default_output_dir="my_data/search-tool-star",
        default_train_source="my_data/tmp/raw/tool_star/toolstar30k.json",
        default_test_source="my_data/tmp/raw/tool_star/toolstar30k.json",
    )
    args = parser.parse_args()

    convert_dataset(
        train_source=args.train_data_source,
        test_source=args.test_data_source,
        output_dir=args.local_dir,
        make_map_fn=make_map_fn,
    )
