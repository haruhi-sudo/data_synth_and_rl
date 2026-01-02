<h1 align="center" style="margin-top: -50px;">Mock Worlds, Real Skills: Building Small Agentic Language Models with Synthetic Tasks, Simulated Environments, and Rubric-Based Rewards</h1>

# 📑 Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
  - [Data Synthesis](#data-synthesis)
  - [RL training](#rl-training)
- [Inference](#inference)
- [Project Structure](#project-structure)

# Overview

This project presents a framework for building small agentic language models using synthetic tasks, simulated environments, and rubric-based rewards. The framework consists of two main components:

1. **Data Synthesis Module** (`tool_use_data_synthesis`): Generates synthetic tool-use tasks and rubrics using big LLMs to train smaller, more efficient agents.

2. **Reinforcement Learning Training Module** (`RL`): Implements reinforcement learning with a custom library (verl) to train language models on the synthetic data using rubric-based reward systems.

# Quick Start

## Prerequisites

- Python 3.10+
- Required Python packages (langgraph, openai, dotenv)
- Access to an LLM API (configured in .env and YAML files)

## Data Synthesis

The data synthesis module generates synthetic tool-use data by:

1. Creating virtual tasks and tool sets via Personas
2. Try to solve tasks multiple times
3. Creating rubrics for each task

Run task synthesis:
```bash
cd tool_use_data_synthesis
python tool_use_data_gen.py --config configs/tool_use_data_gen.yaml
```

Then, attempt to solve tasks:
```bash
python solve_task.py --config configs/solve_task.yaml
```

Finally, run rubric synthesis:
```bash
python rubrics.py --config configs/rubrics.yaml
```

Note: We have provided partial synthetic data(including synthetic tasks and rubrics) in [RL/my_data/virtual_tool_use](RL/my_data/virtual_tool_use).

## RL Training
The RL training module uses the `verl` library. Follow the **[Verl sglang worker installation guide](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html#installation)** to set up the RL environment.  

Before training on the virtual tool-use tasks, it is highly recommended to training on the reasoning tasks first.
```bash
cd RL/
bash my_script/scripts/run_reasoning.sh
```
Reasoning tasks contain math and search tasks. Since search tasks extensive web searches and are costly, we provide a script for only math tasks for easy reproduction.

Then Run the virtual tool-use RL training:
```bash
cd RL/
bash my_script/scripts/run_virtual_tool.sh
```


Note: For your customized need, modify [data processing scripts](RL/my_script/data_process), our implementation of the [python interpreter and web search tools](RL/my_script/tools) and the [script](RL/my_script/scripts/run_virtual_tool.sh).

# Inference
After training, refer to the official [BFCL](https://github.com/ShishirPatil/gorilla) or [TAU2](https://github.com/sierra-research/tau2-bench) benchmark for evaluation. Remember, we train on non-thinking mode and our customized prompts, so you need to modify the prompts and the evaluation script.

# Project Structure
```
synthagent/
├── tool_use_data_synthesis/      # Data generation module
│   ├── configs/                  # Configuration files
│   │   ├── rubrics.yaml          # Rubrics generation settings
│   │   ├── solve_task.yaml       # Task solving configuration
│   │   └── tool_use_data_gen.yaml # Data generation settings
│   ├── functions/                # Core functions
│   │   ├── call_llms.py          # LLM interaction utilities
│   │   ├── fuzzy_task.py         # Task fuzzification
│   │   ├── mock_tools.py         # Mock tool implementations
│   │   ├── mock_user.py          # User simulation
│   │   ├── solve_task.py         # Task solving logic
│   │   ├── tool_check.py         # Tool validation
│   │   └── tool_set_gen.py       # Tool set generation
│   ├── graph/                    # Agent workflow graphs
│   │   ├── graph_solve_task.py   # Task solving graph
│   │   └── graph_virtual_tools.py # Virtual tools graph
│   ├── output/                   # Generated data output
│   ├── configuration.py          # Configuration utilities
│   ├── rubrics.py                # Rubric generation
│   ├── solve_task.py             # Solve the synthetic tasks
│   └── tool_use_data_gen.py      # Main data generation script
├── RL/                           # Reinforcement learning module
│   ├── my_script/                # Custom RL scripts
│   │   ├── data_process/         # Data processing utilities
│   │   ├── scripts/              # Training scripts
│   │   ├── tools/                # Tool implementations
│   │   ├── reward_function.py    # Reward calculation
│   │   └── tool_config.yaml      # Tool configuration
│   ├── verl/                     # VERL (Volcano Engine RL) library
|   |   |
│   │   ├── experimental/agent_loop/tool_agent_loop.py         # We modify this file to add our custom agent loop(my_tool_agent_with_user)
│   │   ├── experimental/agent_loop/tool_parser.py             # We modify this file to add our custom tool parser(my_custom_hermes)
│   └── checkpoints/              # Model checkpoints
└── README.md                     # This file
```