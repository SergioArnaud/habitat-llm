#!/bin/bash
## SLURM scripts have a specific format. 

#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=10
#SBATCH --open-mode=append
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --signal=USR1@60

# setup conda and shell environments
module purge
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate habitat-llm

# Setup slurm multinode
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR
set -x
#export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES && export HABITAT_ENV_DEBUG=1
echo 1

srun python -u habitat-lab/habitat-baselines/habitat_baselines/run.py \
    --exp-config habitat-lab/habitat-baselines/habitat_baselines/config/rearrange/rl_skill.yaml \
    --run-type eval \
    habitat_baselines.checkpoint_folder=$3/checkpoints \
    habitat_baselines.eval_ckpt_path_dir=$3/checkpoints \
    habitat_baselines.video_dir=$3/video_dir \
    benchmark/rearrange=$1 habitat_baselines.wb.run_name=$2 habitat_baselines.wb.group=$2 #\
    #habitat/simulator/agents@habitat.simulator.agents.main_agent=depth_head_agent_vis
