#!/bin/bash

set -exo pipefail

export VLLM_ATTENTION_BACKEND=XFORMERS

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

yaml_file=${1}
shift 1

redaccel-cli rl ${yaml_file}\
             data.data_source=rec_old \
             algorithm.adv_estimator=grpo \
             data.train_batch_size=8 \
             data.val_batch_size=8 \
             data.max_prompt_length=2048 \
             data.max_response_length=1024 \
	     data.truncation=left \
             actor_rollout_ref.actor.optim.lr=1e-6 \
             actor_rollout_ref.model.use_remove_padding=True \
             actor_rollout_ref.actor.ppo_mini_batch_size=8 \
             actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
             actor_rollout_ref.actor.use_kl_loss=True \
             actor_rollout_ref.actor.kl_loss_coef=0.001 \
             actor_rollout_ref.actor.kl_loss_type=low_var_kl \
             actor_rollout_ref.model.enable_gradient_checkpointing=True \
             actor_rollout_ref.actor.fsdp_config.param_offload=False \
             actor_rollout_ref.actor.fsdp_config.grad_offload=False \
             actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
             actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
             actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
             actor_rollout_ref.rollout.name=vllm \
             actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
             actor_rollout_ref.rollout.n=5 \
             actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
             actor_rollout_ref.ref.fsdp_config.param_offload=True \
             algorithm.kl_ctrl.kl_coef=0.001 \
             trainer.critic_warmup=0 \
             trainer.n_gpus_per_node=4 \
             trainer.nnodes=1 \
             trainer.save_freq=500 \
             trainer.test_freq=50 \
             trainer.total_epochs=1 $@



	     #custom_reward_function.path=/mnt/ali-sh-1/usr/zanghai1/workplace/red_verl/redaccelexamples/plugins/grpo \
             #custom_reward_function.name=compute_score \
             #reward_model.reward_manager=naive \
