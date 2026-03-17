#!/bin/bash
# run on 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

# Launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

N_NODES=2
N_GPUS_PER_NODE=8

rank=${RANK:-0}
if [ "$rank" -eq 0 ]; then
    echo "Executing ray start --head on master node (rank=0)..."
    ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${N_GPUS_PER_NODE} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
else
    echo "Executing ray start --address on worker node (rank=${rank})..."
    ray start --address=${MASTER_ADDR}:6379 --num-gpus ${N_GPUS_PER_NODE} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --block
fi

# Only submit the job from the master node
if [ "$rank" -ne 0 ]; then
    echo "Worker node (rank=${rank}) will block on ray start. Exiting..."
    exit 0
fi

PROJECT_DIR="$(pwd)"
TOOL_CONFIG_PATH="$PROJECT_DIR/my_script/tool_config.yaml"

PROJECT_NAME="qwen30b"
EXPERIMENT_NAME="test"

# Hyperparameters
TRAIN_BATCH_SIZE=64
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=10000
PPO_MINI_BATCH_SIZE=16
PPO_MICRO_BATCH_SIZE_PER_GPU=4
LR=1e-6
MAX_TURNS=15
clip_ratio_low=0.0003
clip_ratio_high=0.0004
ROLLOUT_N=16

# Model path and save path
BEFOREPOINTS="..."
AFTERPOINTS="checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME"

# Data
TRAIN_FILES="my_data/tmp/tool_use_data_filtered/train.parquet"
VAL_FILES="my_data/tmp/tool_use_data_filtered/val.parquet"

python3.10 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${BEFOREPOINTS}  \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.format=my_custom_hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4096 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_TURNS \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG_PATH} \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${AFTERPOINTS} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${N_NODES} \
    trainer.save_freq=6 \
    trainer.test_freq=3 \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    trainer.total_epochs=1 \
    reward_model.reward_manager=dapo \
    custom_reward_function.path=my_script/reward_function.py \
    custom_reward_function.name=compute_score_virtual_tool_completion \

# Auto-detect the latest checkpoint step
LATEST_STEP=$(cat ${AFTERPOINTS}/latest_checkpointed_iteration.txt 2>/dev/null || echo "")
if [ -z "$LATEST_STEP" ]; then
    echo "Warning: Could not find latest_checkpointed_iteration.txt, skipping merge"
else
    echo "Merging checkpoint from global_step_${LATEST_STEP}"
    python3.10 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir ${AFTERPOINTS}/global_step_${LATEST_STEP}/actor \
        --target_dir ${AFTERPOINTS}/merge
fi
