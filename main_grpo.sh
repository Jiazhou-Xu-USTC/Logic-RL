#!/usr/bin/env bash
set -x
set -o errexit
set -o pipefail

# 1. Clean Conda cache safely
if command -v conda &> /dev/null; then
    conda clean --all --yes
else
    rm -rf /root/miniconda3/pkgs/*
fi

# 2. Clean HuggingFace cache files older than 7 days
find /root/autodl-tmp/hf_cache -type f -mtime +7 -exec rm -f {} \; || true

# 3. Clean old checkpoints: keep only the 2 most recent (adjust as needed)
CKPT_DIR="/root/autodl-tmp/checkpoints/actor"
if [ -d "$CKPT_DIR" ]; then
    cd "$CKPT_DIR"
    ls -dt global_step_* 2>/dev/null | tail -n +3 | xargs rm -rf
    cd -
fi

# 4. Clean temp files older than 2 days
find /root/autodl-tmp -type f -mtime +2 -exec rm -f {} \; || true
find /tmp -type f -mtime +2 -exec rm -f {} \; || true

# 5. Clean trash (user and root)
rm -rf ~/.local/share/Trash/* /.Trash-0/* || true

# 6. Set environment variables for HuggingFace and CUDA
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_TIMEOUT=120

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=1

MODEL_PATH=Qwen/Qwen2.5-7B-Instruct-1M
export VLLM_ATTENTION_BACKEND=XFORMERS

# 7. Run Logic-RL training
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files=data/kk/instruct/3ppl/train.parquet \
    data.val_files=data/kk/instruct/3ppl/test.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    data.max_prompt_length=400 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
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
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='reinforce_plus_plus_logic_KK' \
    trainer.experiment_name='Qwen-7B' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir=/root/autodl-fs/checkpoints \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 $@ 2>&1 | tee grpo.log
