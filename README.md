<h1 align="center">AgenticQwen: Building Small Agentic Language Models with Synthetic Data and Multi-Round Reinforcement Learning</h1>

# Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
  - [Data Synthesis](#data-synthesis)
  - [RL Training](#rl-training)
- [Inference](#inference)
- [Project Structure](#project-structure)

# Overview

This repository contains the **agentic RL** data synthesis and training code for **AgenticQwen**, a family of small agentic language models built on Qwen backbones. 

This repo focuses on the **agentic RL** pipeline: the model interacts with simulated users and virtual tool environments targeting real-world scenarios (e.g., booking flights, managing accounts), and receives rewards from rubric-based evaluators.

**Data generation with behavior trees.**  The data synthesis module can directly generate multi-branch behavior-tree-structured tasks from persona backgrounds (no linear-solution initialization is required). This differs from the paper's iterative data flywheel, where linear workflows are gradually expanded into behavior trees over multiple RL rounds. We make this simplification for ease of use. If you want to replicate the flywheel and expand an existing workflow into a richer behavior tree, pass the workflow via the `normal_workflow` field in `seed_info` (see [`graph/virtual_tools.py` L53-56](tool_use_data_synthesis/graph/virtual_tools.py)).

The repository is organized into two main modules:

- **`tool_use_data_synthesis/`** -- Generates synthetic agentic tasks, tree-based policies, and rubrics using LangGraph-based pipelines driven by large LLMs.
- **`RL/`** -- Implements multi-round RL training with the [verl](https://github.com/volcengine/verl) library and SGLang-based rollout, including custom reward functions and simulated tool/user environments.

# Quick Start

## Prerequisites

- Python 3.10
- Required Python packages (`langgraph`, `openai`, `python-dotenv`)
- Access to an LLM API (configured in `.env` and YAML files)

## Data Synthesis

The data synthesis module generates synthetic agentic RL training data through a three-stage pipeline:

**Stage 1: Task and Tool Generation** -- Generate virtual tool sets, policies, and tasks from persona backgrounds:
```bash
cd tool_use_data_synthesis
python run_data_gen.py --config configs/data_gen.yaml
```

**Stage 2: Task Solving** -- Attempt to solve each generated task using a large model with simulated tools and users:
```bash
python run_solve_task.py --config configs/solve_task.yaml
```

**Stage 3: Rubric Evaluation** -- Evaluate solution trajectories and filter the data for subsequent training:
```bash
python run_rubrics.py --config configs/rubrics.yaml

# collect training data
python make_filtered_verl_data.py
```

## RL Training

The RL training module uses the `verl` library with SGLang for rollout. 
```
cd RL/
USE_MEGATRON=0 bash my_script/scripts/install.sh
pip install --no-deps .
```
Follow the **[verl SGLang worker installation guide](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html#installation)** to set up the environment.


**(Optional) Reasoning RL** -- If you wish to warm up the model with reasoning tasks before agentic RL, a script is provided for math tasks:
```bash
cd RL/
bash my_script/scripts/run_reasoning.sh
```

**Agentic RL** -- Train on virtual tool-use tasks with simulated users and environments:
```bash
cd RL/
bash my_script/scripts/run_virtual_tool.sh
```

Note: For customization, modify the [data processing scripts](RL/my_script/data_process), the [tool implementations](RL/my_script/tools) (mock tool, python interpreter, web search), and the [training scripts](RL/my_script/scripts/).

# Inference

After training, refer to the official [BFCL](https://github.com/ShishirPatil/gorilla) or [TAU-bench](https://github.com/sierra-research/tau2-bench) benchmarks for evaluation. Note that we train in non-thinking mode with customized prompts, so you may need to adapt the prompts and evaluation scripts accordingly.

# Project Structure
```
AgenticQwen/
├── tool_use_data_synthesis/           # Synthetic data generation module
│   ├── configs/                       # Configuration files
│   │   ├── data_gen.yaml              # Task & tool generation settings
│   │   ├── solve_task.yaml            # Task solving configuration
│   │   ├── rubrics.yaml               # Rubric evaluation settings
│   │   └── persona_5K.jsonl           # 5K persona backgrounds for task generation
│   ├── functions/                     # Core logic
│   │   ├── tool_set_policy_gen.py     # Tool set and policy tree generation
│   │   ├── policy_task.py             # Policy-to-natural-language and test case generation
│   │   ├── refine_policy_task.py      # Task refinement, user backgrounds, hack paths
│   │   ├── solve_task.py              # LLM-based task solving with tool calls
│   │   ├── mock_tools.py             # LLM-simulated tool execution
│   │   ├── mock_user.py              # Simulated user responses
│   │   └── call_llms.py              # LLM API wrapper
│   ├── graph/                         # LangGraph workflow definitions
│   │   ├── virtual_tools.py           # Data generation graph (toolset -> policy -> task)
│   │   └── solve_task.py              # Task solving graph (agent <-> tools/user loop)
│   ├── output/                        # Generated data output
│   ├── run_data_gen.py                # Entry: generate tasks and tools
│   ├── run_solve_task.py              # Entry: solve generated tasks
│   ├── run_rubrics.py                 # Entry: evaluate solutions and produce rubrics
│   ├── make_filtered_verl_data.py     # Entry: training data collection
│   └── configuration.py               # Model configuration utilities
├── RL/                                # Reinforcement learning module
│   ├── my_script/                     # Custom RL scripts
│   │   ├── scripts/                   # Training launch scripts
│   │   │   ├── run_reasoning.sh       # Reasoning RL (math + search)
│   │   │   └── run_virtual_tool.sh    # Agentic RL (virtual tool-use)
│   │   ├── data_process/              # Data conversion to verl format
│   │   │   ├── virtual_tool_use_convert_parquet.py
│   │   │   ├── reasoning_convert_parquet.py
│   │   │   └── base.py
│   │   ├── tools/                     # Tool implementations for RL rollout
│   │   │   ├── mock_tool.py           # LLM-simulated tool and user for agentic RL
│   │   │   ├── search_tool.py         # Web search tool for reasoning RL
│   │   │   └── python_tool.py         # Python code interpreter
│   │   ├── prompts/                   # Prompt templates
│   │   │   ├── judge.py               # Rubric-based judging prompts
│   │   │   ├── mock_user.py           # Mock user simulation prompt
│   │   │   └── tool_simulation.py     # Tool simulation prompt
│   │   ├── utils/                     # Shared utilities
│   │   │   ├── llm_client.py
│   │   │   └── message_parser.py
│   │   ├── reward_function.py         # Reward functions (rubric-based and math)
│   │   └── tool_config.yaml           # Tool registration for RL rollout
│   ├── my_data/                       # Training and validation data
│   │   └── raw/                       # Pre-converted JSON data
│   ├── verl/                          # verl (Volcano Engine RL) library
│   │   ├── experimental/agent_loop/
│   │   │   ├── tool_agent_loop.py     # Custom agent loop with user interaction
│   │   │   └── tool_parser.py         # Custom tool parser (tool_call, question, answer)
│   └── checkpoints/                   # Model checkpoints
└── README.md
```
