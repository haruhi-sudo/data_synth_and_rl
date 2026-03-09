"""
Filter solved tasks by rubrics evaluation (PASS only) and produce verl training data.

Usage:
    python dev_tools/make_verl_data_filtered.py \
        --solve_path output/solve_tool_use_235b \
        --output policy_tool_use_data_filtered.json
"""
import os
import json
import random
import argparse


def main():
    parser = argparse.ArgumentParser(description="Filter PASS solutions into verl training data")
    parser.add_argument("--solve_path", type=str, default="output/solve_tool_use_235b",
                        help="Path to the solve output directory")
    parser.add_argument("--output", type=str, default="tool_use_data_filtered.json",
                        help="Output JSON file path")
    parser.add_argument("--val_split", type=int, default=100,
                        help="Number of samples to hold out for validation (0 to disable)")
    args = parser.parse_args()

    solve_path = args.solve_path
    saves = []
    skipped_no_rubrics = 0
    skipped_no_pass = 0

    for task_id in sorted(os.listdir(solve_path)):
        task_dir = os.path.join(solve_path, task_id)
        if not os.path.isdir(task_dir):
            continue

        rubrics_path = os.path.join(task_dir, "rubrics_output.json")
        more_info_path = os.path.join(task_dir, "more_info.json")

        if not os.path.exists(rubrics_path) or not os.path.exists(more_info_path):
            skipped_no_rubrics += 1
            continue

        with open(rubrics_path, 'r', encoding='utf-8') as f:
            rubrics_output = json.load(f)

        if rubrics_output.get("pass_count", 0) == 0:
            skipped_no_pass += 1
            continue

        best_solution_name = rubrics_output.get("best_solution")
        if not best_solution_name:
            skipped_no_pass += 1
            continue

        best_solution_path = os.path.join(task_dir, best_solution_name)
        if not os.path.exists(best_solution_path):
            skipped_no_pass += 1
            continue

        with open(best_solution_path, 'r', encoding='utf-8') as f:
            solution = json.load(f)

        with open(more_info_path, 'r', encoding='utf-8') as f:
            more_info = json.load(f)

        saves.append({
            "id": task_id,
            "prompt": solution[:2],
            "task_background": more_info.get("user_background", ""),
            "rubrics": more_info.get("evaluation", ""),
            "test_policy": more_info.get("test_policy", ""),
            "user_escape_strategy": more_info.get("user_escape_strategy", ""),
            "tool_return_expected": more_info.get("tool_return_expected", {}),
        })

    print(f"Total PASS: {len(saves)}, skipped (no rubrics): {skipped_no_rubrics}, skipped (no PASS): {skipped_no_pass}")

    random.shuffle(saves)

    if args.val_split > 0 and len(saves) > args.val_split:
        val_data = saves[:args.val_split]
        train_data = saves[args.val_split:]

        val_output = args.output.replace(".json", "_val.json")
        with open(val_output, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=4)
        print(f"Validation set: {len(val_data)} -> {val_output}")
    else:
        train_data = saves

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    print(f"Training set: {len(train_data)} -> {args.output}")


if __name__ == "__main__":
    main()
