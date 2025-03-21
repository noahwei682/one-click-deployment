conda create -n llava-next python=3.10 -y
source activate llava-next
conda activate llava-next
pip install --upgrade pip  # Enable PEP 660 support.


git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
# git clone https://ghproxy.net/https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval
pip install -e .

git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
# git clone https://ghproxy.net/https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT

pip install -e ".[train]"
pip install flash-attn==2.5.2 --no-build-isolation

export HF_TOKEN=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
# export HF_HOME=/root/autodl-tmp/huggingface   (本地较大的路径)
# export HF_HOME=/data3/szs/hz/
huggingface-cli login --token $HF_TOKEN

# export MASTER_ADDR="l192.168.1.100"  # 设置主节点地址
# export MASTER_PORT=1235  

# python3 -m accelerate.commands.launch \
#     --num_processes=7 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
#     --tasks scienceqa_img,gqa,mmbench_en,mmbench_cn,pope,llava_in_the_wild,mme,mmvet,seedbench,vizwiz_vqa \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix scienceqa_img,gqa,mmbench_en,mmbench_cn,pope,llava_in_the_wild,mme,mmvet,seedbench,vizwiz_vqa \
#     --output_path ./logs/ \
#     --verbosity=DEBUG \
#     --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 



python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/home/aiscuser/lmms-eval/llava-ov-ewc-grpo/output_dir/checkpoints/gsm8k-llava-onevision-qwen2-7b-ov-grpo/checkpoint-500" \
    --tasks scienceqa_img \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix scienceqa_img_grpo500 \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/home/aiscuser/lmms-eval/llava-ov-ewc-grpo/output_dir/checkpoints/gsm8k-llava-onevision-qwen2-7b-ov-grpo/checkpoint-500" \
    --tasks gqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix gqa_chat_grpo500 \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/home/aiscuser/lmms-eval/llava-ov-ewc-grpo/output_dir/checkpoints/gsm8k-llava-onevision-qwen2-7b-ov-grpo/checkpoint-500" \
    --tasks mmbench_en \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix mmbench_en_grpo500 \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/home/aiscuser/lmms-eval/llava-ov-ewc-grpo/output_dir/checkpoints/gsm8k-llava-onevision-qwen2-7b-ov-grpo/checkpoint-500" \
    --tasks mmbench_cn \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix mmbench_cn_grpo500 \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/home/aiscuser/lmms-eval/llava-ov-ewc-grpo/output_dir/checkpoints/gsm8k-llava-onevision-qwen2-7b-ov-grpo/checkpoint-500" \
    --tasks pope \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix pope_grpo500 \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 



python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/home/aiscuser/lmms-eval/llava-ov-ewc-grpo/output_dir/checkpoints/gsm8k-llava-onevision-qwen2-7b-ov-grpo/checkpoint-500" \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix mme_grpo500 \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 





python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/home/aiscuser/lmms-eval/llava-ov-ewc-grpo/output_dir/checkpoints/gsm8k-llava-onevision-qwen2-7b-ov-grpo/checkpoint-500" \
    --tasks seedbench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix seedbench_grpo500 \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 




python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/home/aiscuser/lmms-eval/llava-ov-ewc-grpo/output_dir/checkpoints/gsm8k-llava-onevision-qwen2-7b-ov-grpo/checkpoint-500" \
    --tasks vizwiz_vqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix vizwiz_vqa_grpo500 \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 
