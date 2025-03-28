conda create -n llava-next python=3.10 -y
source activate llava-next
conda activate llava-next
pip install --upgrade pip  # Enable PEP 660 support.

git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
pip install -e .

git clone https://github.com/noahwei682/llava-ov-ewc-ms-grpo.git
cd llava-ov-ewc-ms-grpo/

pip install -e ".[train]"
pip install flash-attn==2.5.2 --no-build-isolation
pip install ipdb


pip install huggingface_hub
export HF_TOKEN=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
huggingface-cli login --token $HF_TOKEN
# local-dir=/home/aiscuser
conda install -c conda-forge wandb --yes
pip install pydantic --upgrade
wandb login a0686d210ceba8f713f6cd85c5dcf3621b7f15e7


mkdir msdata
# huggingface-cli download wei682/amazon images_20250321.tar.gz --repo-type dataset  --local-dir ./msdata
huggingface-cli download laolao77/ViRFT_CLS_flower_4_shot --repo-type dataset  --local-dir ./msdata
# laolao77/ViRFT_CLS_flower_4_shot
# laolao77/ViRFT_CLS_fgvc_aircraft_4_shot
# laolao77/ViRFT_CLS_car196_4shot
# laolao77/ViRFT_CLS_pets37_4shot


# cd msdata
# mkdir images
# tar -xvzf images_20250321.tar.gz -C ./images

# huggingface-cli download wei682/amazon meta_Electronics_exist_gene_ITdata.json --repo-type dataset  --local-dir .
# huggingface-cli download wei682/amazon meta_Arts_Crafts_and_Sewing_exist_gene_ITdata.json --repo-type dataset  --local-dir .
# huggingface-cli download wei682/amazon meta_Industrial_and_Scientific_exist_gene_ITdata.json --repo-type dataset  --local-dir .
# huggingface-cli download wei682/amazon meta_Health_and_Household_exist_gene_ITdata.json --repo-type dataset  --local-dir .
# huggingface-cli download wei682/amazon meta_Movies_and_TV_exist_gene_ITdata.json --repo-type dataset  --local-dir .
# huggingface-cli download wei682/amazon meta_Beauty_and_Personal_Care_exist_gene_ITdata.json --repo-type dataset  --local-dir .
# huggingface-cli download wei682/amazon meta_Electronics_exist_gene_ITdata.json --repo-type dataset  --local-dir .
# huggingface-cli download wei682/amazon meta_Appliances_exist_gene_ITdata.json --repo-type dataset  --local-dir .
# huggingface-cli download wei682/amazon meta_Cell_Phones_and_Accessories_exist_gene_ITdata.json --repo-type dataset --local-dir .
# huggingface-cli download wei682/amazon meta_Grocery_and_Gourmet_Food_exist_gene_ITdata.json --repo-type dataset  --local-dir .
# huggingface-cli download wei682/amazon meta_Sports_and_Outdoors_exist_gene_ITdata.json --repo-type dataset  --local-dir .
# huggingface-cli download wei682/amazon meta_Magazine_Subscriptions_exist_gene_ITdata.json --repo-type dataset  --local-dir .
# huggingface-cli download wei682/amazon meta_Subscription_Boxes_exist_gene_ITdata.json --repo-type dataset  --local-dir .
# huggingface-cli download wei682/amazon meta_Toys_and_Games_exist_gene_ITdata.json --repo-type dataset  --local-dir .
# huggingface-cli download wei682/amazon meta_All_Beauty_exist_gene_ITdata.json --repo-type dataset  --local-dir .
# huggingface-cli download wei682/amazon meta_Patio_Lawn_and_Garden_exist_gene_ITdata.json --repo-type dataset  --local-dir .
# huggingface-cli download wei682/amazon meta_Gift_Cards_exist_gene_ITdata.json --repo-type dataset  --local-dir .
# cd ..

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export ACCELERATE_CPU_AFFINITY=1 
export NPROC_PER_NODE=4
export NODES=1 
export NODE_RANK=0 

export RUN_NAME="llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewcno-lambda0-ViRFT_CLS_flower_4_shot_debug"
export OUTPUT_DIR="/blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewcno-lambda0-ViRFT_CLS_flower_4_shot_debug"
# export DATA_DIR="/home/aiscuser/lmms-eval/llava-ov-ewc-ms/msdata/"
export PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-7b-si"

bash ./scripts/train/finetune_grpo.sh

sudo apt-get install jq
# jq '. + {"vocab_size": 152064}' /blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewc-lambda01-amazon-multisource-all/config.json > temp.json && mv temp.json /blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewc-lambda01-amazon-multisource-all/config.json
jq '. + {"vocab_size": 152064}' /blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewcno-lambda0-ViRFT_CLS_flower_4_shot/config.json > temp.json && mv temp.json /blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewcno-lambda0-ViRFT_CLS_flower_4_shot/config.json

huggingface-cli upload LLaVA_checkpoint /blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewcno-lambda0-ViRFT_CLS_flower_4_shot ./grpo-ViRFT_CLS_flower_4_shot --repo-type dataset

python3 -m accelerate.commands.launch --num_processes 4 -m lmms_eval --model llava --model_args pretrained=/blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewcno-lambda0-ViRFT_CLS_flower_4_shot/checkpoint-100 --tasks scienceqa_img --batch_size 1 --log_samples --log_samples_suffix scienceqa_img-ViRFT_CLS_flower_4_shot --output_path ./logs/ --verbosity=DEBUG --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 
python3 -m accelerate.commands.launch --num_processes 4 -m lmms_eval --model llava --model_args pretrained=/blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewcno-lambda0-ViRFT_CLS_flower_4_shot/checkpoint-100 --tasks gqa --batch_size 1 --log_samples --log_samples_suffix gqa-ViRFT_CLS_flower_4_shot --output_path ./logs/ --verbosity=DEBUG --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 
python3 -m accelerate.commands.launch --num_processes 4 -m lmms_eval --model llava --model_args pretrained=/blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewcno-lambda0-ViRFT_CLS_flower_4_shot/checkpoint-100 --tasks mmbench_en --batch_size 1 --log_samples --log_samples_suffix mmbench_en-ViRFT_CLS_flower_4_shot --output_path ./logs/ --verbosity=DEBUG --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 
python3 -m accelerate.commands.launch --num_processes 4 -m lmms_eval --model llava --model_args pretrained=/blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewcno-lambda0-ViRFT_CLS_flower_4_shot/checkpoint-100 --tasks mmbench_cn --batch_size 1 --log_samples --log_samples_suffix mmbench_cn-ViRFT_CLS_flower_4_shot --output_path ./logs/ --verbosity=DEBUG --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False
python3 -m accelerate.commands.launch --num_processes 4 -m lmms_eval --model llava --model_args pretrained=/blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewcno-lambda0-ViRFT_CLS_flower_4_shot/checkpoint-100 --tasks pope --batch_size 1 --log_samples --log_samples_suffix pope-ViRFT_CLS_flower_4_shot --output_path ./logs/ --verbosity=DEBUG --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 
python3 -m accelerate.commands.launch --num_processes 4 -m lmms_eval --model llava --model_args pretrained=/blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewcno-lambda0-ViRFT_CLS_flower_4_shot/checkpoint-100 --tasks mme --batch_size 1 --log_samples --log_samples_suffix mme-ViRFT_CLS_flower_4_shot --output_path ./logs/ --verbosity=DEBUG --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 
python3 -m accelerate.commands.launch --num_processes 4 -m lmms_eval --model llava --model_args pretrained=/blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewcno-lambda0-ViRFT_CLS_flower_4_shot/checkpoint-100 --tasks seedbench --batch_size 1 --log_samples --log_samples_suffix seedbench-ViRFT_CLS_flower_4_shot --output_path ./logs/ --verbosity=DEBUG --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 
python3 -m accelerate.commands.launch --num_processes 4 -m lmms_eval --model llava --model_args pretrained=/blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewcno-lambda0-ViRFT_CLS_flower_4_shot/checkpoint-100 --tasks vizwiz_vqa --batch_size 1 --log_samples --log_samples_suffix vizwiz_vqa-ViRFT_CLS_flower_4_shot --output_path ./logs/ --verbosity=DEBUG --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 
