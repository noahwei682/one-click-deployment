conda create -n llava-next python=3.10 -y
source activate llava-next
pip install --upgrade pip  # Enable PEP 660 support.

git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
pip install -e .

git clone https://github.com/noahwei682/LLaVA-NeXT.git
cd LLaVA-NeXT

pip install -e ".[train]"
pip install flash-attn==2.5.2 --no-build-isolation

git clone https://github.com/noahwei682/download_data.git
cd download_data

conda install -c conda-forge wandb --yes
wandb login a0686d210ceba8f713f6cd85c5dcf3621b7f15e7

mkdir mydatasets
cd mydatasets
mkdir llava_onevision
cd llava_onevision
mkdir images
cd images
mkdir FigureQA_MathV360K
cd ..
cd ..
cd ..
python preprocess_llava_onevision_parquet.py
cd ..
mkdir output_dir
cd output_dir
mkdir checkpoints
cd checkpoints
mkdir onevision
cd ..
cd ..

conda install -c conda-forge wandb --yes
wandb login a0686d210ceba8f713f6cd85c5dcf3621b7f15e7

export HF_TOKEN=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
# export HF_HOME=/root/autodl-tmp/huggingface
huggingface-cli login --token $HF_TOKEN

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=br-intranet
export NCCL_DEBUG=INFO

export ACCELERATE_CPU_AFFINITY=1 
export NPROC_PER_NODE=8 
export NODES=1 
export NODE_RANK=0 
export MASTER_ADDR=172.17.100.112 
export MASTER_PORT=23456 

bash ./scripts/train/finetune_grpo_gsm8k.sh
