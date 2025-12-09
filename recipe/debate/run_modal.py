import re
import subprocess
from pathlib import Path
from typing import Literal, Optional

import modal

app = modal.App("debate-verl")

VERL_PATH = "/root/verl"

image = (
    modal.Image.from_registry("vllm/vllm-openai")
    .run_commands("ln -s $(which python3) /usr/bin/python")
    .apt_install("git")
    # .uv_pip_install("verl[sglang]==0.6.1")
    .uv_pip_install(
        "accelerate",
        "codetiming",
        "datasets",
        "dill",
        "hydra-core",
        "numpy<2.0.0",
        "pandas",
        "peft",
        "pyarrow>=19.0.0",
        "pybind11",
        "pylatexenc",
        "ray[default]>=2.10",
        "tensordict>=0.8.0,<=0.10.0,!=0.9.0",
        "torchdata",
        "torchvision",  
        "transformers",
        "wandb",
        "huggingface_hub"
    ) # vllm already installed ?
    .add_local_dir("~/code_desynced/verl-self-play/", VERL_PATH)
)


DEBATE_ROOT_PATH: Path = Path(VERL_PATH + "/recipe/debate")

PATH_TO_REWARD_FUNCTION: Path = DEBATE_ROOT_PATH / "debate_reward.py"
REWARD_FUNCTION_NAME: str = "compute_score_debate"
REWARD_LOOP_MANAGER_NAME: str = "debate_reward_loop_manager"
MODELS_PATH: Path = Path("/models")
MINUTES: int = 60

checkpoints_volume: modal.Volume = modal.Volume.from_name(
    "debate-verl-checkpoints", create_if_missing=True
)

"""
update epochs and model path
"""


@app.function(
    image=image,
    gpu="A100-80GB:8",
    volumes={
        MODELS_PATH: checkpoints_volume
    },
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
    timeout=60 * MINUTES,#24 * 60 * MINUTES,
    scaledown_window = 2 * MINUTES,
    env={
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HYDRA_FULL_ERROR": "1",
        "VLLM_USE_V1": "1"
    },
)
def train(*arglist) -> None:
    import os
    cmd: list[str] = [
        "python",
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo_self_play",
        f"data.train_files={DEBATE_ROOT_PATH / 'train.parquet'}",
        f"data.val_files={DEBATE_ROOT_PATH / 'test.parquet'}",
        "data.train_batch_size=16",
        "data.max_prompt_length=128", # 128
        "data.max_response_length=1024",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"data.custom_cls.path={DEBATE_ROOT_PATH / 'vanilla_dataset.py'}",
        "data.custom_cls.name=VanillaDataset",
        "actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507",#meta-llama/Llama-3.1-8B-Instruct",#google/gemma-3-12b-it",
        "actor_rollout_ref.actor.optim.lr=1e-5",
        "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.actor.ppo_mini_batch_size=16",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.actor.checkpoint.save_contents='model,optimizer,extra,hf_model'",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=4",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
        "actor_rollout_ref.rollout.n=5",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/debate/debate_loop_config.yaml",
        "actor_rollout_ref.rollout.dtype=bfloat16",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.rollout.enforce_eager=true",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        "trainer.logger=['console', 'wandb']",
        "trainer.project_name=debate-verl",
        "trainer.experiment_name=small-model-debug",
        "trainer.n_gpus_per_node=4",
        "trainer.nnodes=1",
        "trainer.test_freq=10000000",
        f"trainer.default_local_dir={MODELS_PATH}",
        "trainer.resume_mode=disable",
        # Parameters chosen to ensure easy automated testing. Remove if needed.
        "trainer.save_freq=1",
        #"trainer.total_training_steps=1",
        "trainer.total_epochs=10000",
        "trainer.val_before_train=False",
        f"custom_reward_function.path={str(PATH_TO_REWARD_FUNCTION)}",
        f"custom_reward_function.name={REWARD_FUNCTION_NAME}",
        "reward_model.reward_manager_loop=debate_reward_loop_manager",
        "reward_model.enable=True",
        "reward_model.enable_resource_pool=True",
        "reward_model.n_gpus_per_node=4",
        "reward_model.nnodes=1",
        "reward_model.micro_batch_size=16",
        "reward_model.model.path=Qwen/Qwen3-4B-Instruct-2507",#google/gemma-3-4b-it",
        
        "+reward_model.rollout._target_=verl.workers.config.RolloutConfig",
        "+reward_model.rollout.name=vllm",
        "+reward_model.rollout.dtype=bfloat16",
        "+reward_model.rollout.gpu_memory_utilization=0.5",
        "+reward_model.rollout.enforce_eager=true",
        "+reward_model.rollout.cudagraph_capture_sizes=null",
        "+reward_model.rollout.free_cache_engine=true",
        "+reward_model.rollout.data_parallel_size=1",
        "+reward_model.rollout.expert_parallel_size=1",
        "+reward_model.rollout.tensor_model_parallel_size=4",
        "+reward_model.rollout.max_num_batched_tokens=4096",
        "+reward_model.rollout.max_model_len=null",
        "+reward_model.rollout.max_num_seqs=1024",
        "+reward_model.rollout.load_format=auto",
        "+reward_model.rollout.limit_images=null",
        "+reward_model.rollout.disable_log_stats=true",
        "+reward_model.rollout.skip_tokenizer_init=false",
        "+reward_model.rollout.prompt_length=1024",
        "+reward_model.rollout.response_length=64",
    ]
    if arglist:
        cmd.extend(arglist)

    os.chdir(VERL_PATH)
    subprocess.run(cmd, check=True)