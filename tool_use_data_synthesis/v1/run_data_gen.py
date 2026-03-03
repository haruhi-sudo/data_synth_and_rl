import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import yaml
import concurrent.futures
import argparse
from typing import List, Dict, Any, Set
from v1.graph.virtual_tools import run_agent

def main():
    parser = argparse.ArgumentParser(description='Run v1 data generation tasks from a JSONL file')
    parser.add_argument('--config', type=str, default='v1/configs/data_gen.yaml',
                       help='Path to the configuration file (default: v1/configs/data_gen.yaml)')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    def get_processed_ids(log_file_path: str) -> Set[str]:
        processed_ids = set()
        if not os.path.exists(log_file_path):
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            return processed_ids
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        if "id" in log_entry:
                            processed_ids.add(log_entry["id"])
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Warning: Error reading log file: {e}")
        return processed_ids

    def process_single_task(task_data: Dict[str, Any], config: Dict[str, Any]) -> bool:
        try:
            logger.info(f"Processing task: {task_data['id']}")
            run_agent(
                seed_info={
                    "id": task_data['id'],
                    "background": task_data["persona"]
                },
                run_config=config
            )
            return True
        except Exception as e:
            logger.error(f"Error processing task {task_data['id']}: {e}")
            return False

    def run_tasks_from_file(config: Dict[str, Any]):
        processed_ids = get_processed_ids(config["logging"]["task_file_path"])
        tasks_to_process: List[Dict[str, Any]] = []
        with open(config["paths"]["data_file"], 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    task_data = json.loads(line.strip())
                    if task_data['id'] in processed_ids:
                        logger.info(f"Skipping processed task: {task_data['id']}")
                        continue
                    tasks_to_process.append(task_data)
                    if config["processing"]["max_tasks"] and len(tasks_to_process) >= config["processing"]["max_tasks"]:
                        break
                except json.JSONDecodeError:
                    logger.warning(f"Warning: Skipping invalid JSON line")
                except Exception as e:
                    logger.error(f"Error loading task: {e}")
        if not tasks_to_process:
            logger.info("No new tasks to process")
            return
        logger.info(f"Starting concurrent processing of {len(tasks_to_process)} tasks with {config['processing']['max_workers']} worker threads")
        with concurrent.futures.ThreadPoolExecutor(max_workers=config["processing"]["max_workers"]) as executor:
            future_to_task = {
                executor.submit(process_single_task, task_data, config): task_data
                for task_data in tasks_to_process
            }
            completed_tasks = 0
            failed_tasks = 0
            for future in concurrent.futures.as_completed(future_to_task):
                task_data = future_to_task[future]
                try:
                    success = future.result()
                    if success:
                        completed_tasks += 1
                    else:
                        failed_tasks += 1
                except Exception as e:
                    logger.error(f"Task {task_data['id']} generated an exception: {e}")
                    failed_tasks += 1
            logger.info(f"Processing completed: {completed_tasks} successful, {failed_tasks} failed")

    run_tasks_from_file(config)

if __name__ == "__main__":
    main()
