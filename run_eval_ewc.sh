conda create -n llava-next python=3.10 -y
pip install --upgrade pip  # Enable PEP 660 support.

git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
pip install -e .

git clone https://github.com/noahwei682/llava-ov-ewc.git
cd llava-ov-ewc/

pip install -e ".[train]"
pip install flash-attn==2.5.2 --no-build-isolation
pip install ipdb

git clone https://github.com/noahwei682/download_data.git
cd download_data


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
mkdir huggingface

conda install -c conda-forge wandb --yes
pip install --upgrade wandb
wandb login a0686d210ceba8f713f6cd85c5dcf3621b7f15e7

export HF_TOKEN=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
# export HF_HOME=/data/harold/mhj/huggingface
# export HF_HOME=/scratch
pip install huggingface_hub
huggingface-cli login --token $HF_TOKEN

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_DEBUG=INFO

export ACCELERATE_CPU_AFFINITY=1 
export NPROC_PER_NODE=4 
export NODES=1 
export NODE_RANK=0 
export MASTER_ADDR="10.36.38.89"
export MASTER_PORT="23456"  

# export SERVER_IP="10.36.38.89"
# export SERVER_PORT="23456"
# export NETWORK_INTERFACE="eth0"

# export SERVER_IP="127.0.0.1/8"
# export SERVER_PORT="8080"
# export NETWORK_INTERFACE="lo"


# bash ./scripts/train/finetune_grpo_gsm8k.sh
bash ./scripts/train/finetune_ov_FigureQA_MathV360K.sh

export HF_TOKEN=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
huggingface-cli login --token $HF_TOKEN
pip install huggingface-hub
huggingface-cli upload LLaVA_checkpoint /data/harold/mhj/output_dir/checkpoints/onevision/llava-gsm8k-lmms-lab_llava-onevision-qwen2-7b-ov-grpo ./llava_next/grpo --repo-type dataset

SCRIPT_PATH="./scripts/train/finetune_grpo_gsm8k.sh"
OUTPUT_PREFIX=$(grep "OUTPUT_PREFIX=" "$SCRIPT_PATH" | cut -d'"' -f2)
LLM_VERSION=$(grep "LLM_VERSION=" "$SCRIPT_PATH" | cut -d'"' -f2)
LLM_VERSION_CLEAN=$(echo $LLM_VERSION | sed 's/\//_/g')
RUN_NAME="llava-gsm8k-${LLM_VERSION_CLEAN}-grpo"
OUTPUT_DIR="$OUTPUT_PREFIX/$RUN_NAME/checkpoint-1400"

echo "Output directory will be: $OUTPUT_DIR"



python3 -m accelerate.commands.launch \
    --num_processes=7 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="$OUTPUT_DIR" \
    --tasks scienceqa_img \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix scienceqa_img_base_3_12_run_eval \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=7 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="$OUTPUT_DIR" \
    --tasks gqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix gqa_base_3_12_run_eval \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=7 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="$OUTPUT_DIR" \
    --tasks mmbench_en \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix mmbench_en_base_3_12_run_eval \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=7 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="$OUTPUT_DIR" \
    --tasks mmbench_cn \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix mmbench_cn_base_3_12_run_eval \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=7 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="$OUTPUT_DIR" \
    --tasks pope \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix pope_base_3_12_run_eval \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=7 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="$OUTPUT_DIR" \
    --tasks llava_in_the_wild \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_in_the_wild_base_3_12_run_eval \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=7 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="$OUTPUT_DIR" \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix mme_base_3_12_run_eval \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=7 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="$OUTPUT_DIR" \
    --tasks mmvet \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix mmvet_base_3_12_run_eval \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=7 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="$OUTPUT_DIR" \
    --tasks scienceqa_img,gqa,mmbench_en,mmbench_cn,pope,llava_in_the_wild,mme,mmvet,seedbench,vizwiz_vqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix seedbench_base_3_12_run_eval \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=7 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="$OUTPUT_DIR" \
    --tasks vizwiz_vqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix vizwiz_vqa_base_3_12_run_eval \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 
