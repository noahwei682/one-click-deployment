conda create -n llava-next python=3.10 -y
source activate llava-next
conda activate llava-next
# - pip install --upgrade pip  # Enable PEP 660 support.

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
pip install urllib3
huggingface-cli login --token $HF_TOKEN

# - export OMP_NUM_THREADS=7
export NCCL_IB_DISABLE=1
# - export NCCL_IB_GID_INDEX=3
# - export NCCL_SOCKET_IFNAME="eth0"
export NCCL_DEBUG=INFO
export ACCELERATE_CPU_AFFINITY=1 
export NPROC_PER_NODE=4
export NODES=1 
export NODE_RANK=0 
# export WORLD_SIZE=4
# - export MASTER_ADDR="10.36.38.89"
# - export MASTER_PORT="23456"

export RUN_NAME="llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewc-stage-lambda-1"
export OUTPUT_DIR="/blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewc-stage-lambda-1"
export PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-7b-si"
bash ./scripts/train/finetune_ov_FigureQA_MathV360K.sh
huggingface-cli upload LLaVA_checkpoint /blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewc-stage-lambda-1/checkpoint-10 ./llava-onevision-qwen2-7b-si-ewc-lambda-1 --repo-type dataset
sudo apt-get install jq
>
  jq '. + {"vocab_size": 152064}' \
  /blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewc-stage-lambda-1/checkpoint-10/config.json \
  > /blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewc-stage-lambda-1/checkpoint-10/tmp.json && \
  mv /blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewc-stage-lambda-1/checkpoint-10/tmp.json \
  /blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewc-stage-lambda-1/checkpoint-10/config.json
python3 -m accelerate.commands.launch --num_processes 8 -m lmms_eval --model llava --model_args pretrained=/blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewc-stage-lambda-1/checkpoint-10 --tasks scienceqa_img --batch_size 1 --log_samples --log_samples_suffix scienceqa_img_llava_ov_ewc_lambda1 --output_path ./logs/ --verbosity=DEBUG --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 
python3 -m accelerate.commands.launch --num_processes 8 -m lmms_eval --model llava --model_args pretrained=/blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewc-stage-lambda-1/checkpoint-10 --tasks gqa --batch_size 1 --log_samples --log_samples_suffix gqa_llava_ov_ewc_lambda1 --output_path ./logs/ --verbosity=DEBUG --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 
python3 -m accelerate.commands.launch --num_processes 8 -m lmms_eval --model llava --model_args pretrained=/blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewc-stage-lambda-1/checkpoint-10 --tasks mmbench_en --batch_size 1 --log_samples --log_samples_suffix mmbench_en_llava_ov_ewc_lambda1 --output_path ./logs/ --verbosity=DEBUG --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 
python3 -m accelerate.commands.launch --num_processes 8 -m lmms_eval --model llava --model_args pretrained=/blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewc-stage-lambda-1/checkpoint-10 --tasks mmbench_cn --batch_size 1 --log_samples --log_samples_suffix mmbench_cn_llava_ov_ewc_lambda1 --output_path ./logs/ --verbosity=DEBUG --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False
python3 -m accelerate.commands.launch --num_processes 8 -m lmms_eval --model llava --model_args pretrained=/blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewc-stage-lambda-1/checkpoint-10 --tasks pope --batch_size 1 --log_samples --log_samples_suffix pope_llava_ov_ewc_lambda1 --output_path ./logs/ --verbosity=DEBUG --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 
python3 -m accelerate.commands.launch --num_processes 8 -m lmms_eval --model llava --model_args pretrained=/blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewc-stage-lambda-1/checkpoint-10 --tasks mme --batch_size 1 --log_samples --log_samples_suffix mme_llava_ov_ewc_lambda1 --output_path ./logs/ --verbosity=DEBUG --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 
python3 -m accelerate.commands.launch --num_processes 8 -m lmms_eval --model llava --model_args pretrained=/blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewc-stage-lambda-1/checkpoint-10 --tasks seedbench --batch_size 1 --log_samples --log_samples_suffix seedbench_llava_ov_ewc_lambda1 --output_path ./logs/ --verbosity=DEBUG --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 
python3 -m accelerate.commands.launch --num_processes 8 -m lmms_eval --model llava --model_args pretrained=/blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewc-stage-lambda-1/checkpoint-10 --tasks vizwiz_vqa --batch_size 1 --log_samples --log_samples_suffix vizwiz_vqa_llava_ov_ewc_lambda1 --output_path ./logs/ --verbosity=DEBUG --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 
