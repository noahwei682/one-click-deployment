conda create -n llava-next python=3.10 -y
source activate llava-next
pip install --upgrade pip  # Enable PEP 660 support.

git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
pip install -e .

git clone https://github.com/noahwei682/llava-ov-ewc-grpo.git
cd llava-ov-ewc-grpo

pip install -e ".[train]"
pip install flash-attn==2.5.2 --no-build-isolation

conda install -c conda-forge wandb --yes
pip install pydantic --upgrade
wandb login a0686d210ceba8f713f6cd85c5dcf3621b7f15e7

export HF_TOKEN=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
# export HF_HOME=/data/harold/mhj/huggingface
huggingface-cli login --token $HF_TOKEN


export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

export ACCELERATE_CPU_AFFINITY=1 
export NPROC_PER_NODE=4 
export NODES=1 
export NODE_RANK=0 
export MASTER_ADDR=10.36.37.90
export MASTER_PORT=23456 

# source ~/.bashrc  # 或者 source ~/.bash_profile

bash ./scripts/train/grpo_finetune_gsm8k.sh
