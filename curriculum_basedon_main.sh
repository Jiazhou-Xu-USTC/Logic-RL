#!/usr/bin/env bash
set -x
set -o errexit
set -o pipefail

# Cleanup and environment setup (keep your existing settings)
# ... [existing cleanup commands] ...

# Set core environment variables
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_TIMEOUT=120
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# Curriculum configuration
BASE_CONFIG="\
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    data.max_prompt_length=400 \
    data.max_response_length=2048 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='reinforce_plus_plus_logic_KK' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir=/root/autodl-fs/checkpoints/7ppl \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=60 \
    trainer.test_freq=10"

# Curriculum parameters
PPL_VALUES="7"
INITIAL_MODEL="/root/autodl-fs/checkpoints/actor/global_step_60"
EXPERIMENT_BASE_NAME="Qwen-7B-Curriculum"

# Curriculum loop
MODEL_PATH="$INITIAL_MODEL"
for ppl in $PPL_VALUES; do
    echo "Starting ${ppl}ppl curriculum stage"
    
    CURRENT_EXPERIMENT="${EXPERIMENT_BASE_NAME}-${ppl}ppl"
    TRAIN_FILE="data/kk/instruct/${ppl}ppl/train.parquet"
    VAL_FILE="data/kk/instruct/${ppl}ppl/test.parquet"

    python3 -m verl.trainer.main_ppo \
        ${BASE_CONFIG} \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${VAL_FILE}" \
        actor_rollout_ref.model.path="\"${MODEL_PATH}\"" \
        trainer.experiment_name="${CURRENT_EXPERIMENT}" \
        trainer.total_epochs=5 $@ 2>&1 | tee "${CURRENT_EXPERIMENT}.log"

    # --- Place these lines here, after training and before the next stage ---
    CKPT_DIR="${trainer.default_local_dir}/${ppl}ppl/actor"  # 指向新路径
    LATEST_CKPT=$(ls -d ${CKPT_DIR}/global_step_* | sort -V | tail -n1)
    MODEL_PATH="$LATEST_CKPT"  # 供下一阶段使用
    # ------------------------------------------------------------------------
done

echo "Curriculum training completed!"
