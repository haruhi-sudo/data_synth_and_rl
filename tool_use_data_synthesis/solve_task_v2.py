import json
import logging
import traceback
import yaml
import concurrent.futures
import os
import argparse
from typing import List, Dict, Any, Set

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run math tasks from a JSONL file')
    parser.add_argument('--config', type=str, default='configs/solve_task_v3.yaml',
                       help='Path to the configuration file (default: configs/solve_task_v2.yaml)')
    args = parser.parse_args()

    if args.config == 'configs/solve_task_v3.yaml':
        from graph.graph_solve_task_v3 import run_agent
    else:
        from graph.graph_solve_task_v2 import run_agent

    # Load configuration from YAML file
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    already_processed_file = config["logging"]["already_processed_path"]
    log_dir = os.path.dirname(already_processed_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    def get_processed_ids(log_file_path: str) -> Set[str]:
        """Get a set of already processed task IDs from the log file"""
        processed_ids = set()
        
        # If log file doesn't exist, return empty set
        if not os.path.exists(log_file_path):
            # Create directory path if it doesn't exist
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
                        # Skip invalid lines
                        continue
        except Exception as e:
            logger.warning(f"Warning: Error reading log file: {e}")
        
        return processed_ids

    def process_single_task(task_data: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Process a single task and return True if successful"""
        try:
            logger.info(f"Processing task: {task_data['id']}")
            if not task_data["checked_tools"]:
                return False 

            run_agent(task_data, run_config=config)
            
            return True
        except Exception as e:
            # Print the exact error location (file:line + code line) plus full traceback
            tb = e.__traceback__
            frames = traceback.extract_tb(tb) if tb else []
            # Prefer the deepest frame that points to *our* repo code, not stdlib/site-packages.
            project_root = os.path.dirname(os.path.abspath(__file__))
            user_frames = [
                fr for fr in frames
                if os.path.abspath(fr.filename).startswith(project_root + os.sep)
            ]
            last_user_frame = user_frames[-1] if user_frames else None
            last_frame = frames[-1] if frames else None

            frame_to_report = last_user_frame or last_frame
            if frame_to_report:
                logger.error(
                    "Error processing task %s: %s (at %s:%s in %s: %s)",
                    task_data.get("id", "<unknown>"),
                    e,
                    frame_to_report.filename,
                    frame_to_report.lineno,
                    frame_to_report.name,
                    (frame_to_report.line or "").strip(),
                    exc_info=True,
                )
            else:
                logger.error(
                    "Error processing task %s: %s",
                    task_data.get("id", "<unknown>"),
                    e,
                    exc_info=True,
                )
            return False

    def run_tasks_from_file(config: Dict[str, Any]):
        """Run math tasks from a JSONL file with concurrent processing"""
        # Get already processed IDs
        processed_ids = get_processed_ids(config["logging"]["already_processed_path"])
        tasks_to_process: List[Dict[str, Any]] = []
        
        # Load tasks to process
        with open(config["paths"]["data_file"], 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    task_data = json.loads(line.strip())
                    
                    for idx, task_and_background in enumerate(task_data["tasks_and_backgrounds"]):
                        new_task = {
                            "id": f"{task_data['id']}-{idx}",
                            "policy": task_data["policy"],
                            "task_and_background": task_and_background,
                            "checked_tools": task_data["checked_tools"],
                        }

                        # Skip if already processed
                        if new_task['id'] in processed_ids:
                            logger.info(f"⏭️  Skipping processed task: {new_task['id']}")
                            continue
                    
                        tasks_to_process.append(new_task)
                    
                    # Break if we've reached the max tasks limit
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
        
        # Process tasks concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=config["processing"]["max_workers"]) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(process_single_task, task_data, config): task_data 
                for task_data in tasks_to_process
            }
            
            # Collect results
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
            
            logger.info(f"🏁 Processing completed: {completed_tasks} successful, {failed_tasks} failed")

    # Run with configuration
    run_tasks_from_file(config)

if __name__ == "__main__":
    main()
