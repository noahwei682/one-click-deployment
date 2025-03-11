conda create -n llava-next python=3.10 -y
source activate llava-next
conda activate llava-next
pip install --upgrade pip  # Enable PEP 660 support.

git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
pip install -e .

git clone https://github.com/noahwei682/LLaVA-NeXT.git
cd LLaVA-NeXT

pip install -e ".[train]"
pip install flash-attn==2.5.2 --no-build-isolation

export HF_TOKEN=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
# export HF_HOME=/root/autodl-tmp/huggingface   (本地较大的路径)
huggingface-cli login --token $HF_TOKEN


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
    --num_processes=7 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
    --tasks scienceqa_img \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix scienceqa_img \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=7 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
    --tasks gqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix gqa \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=7 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
    --tasks mmbench_en \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix mmbench_en \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=7 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
    --tasks mmbench_cn \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix mmbench_cn \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=7 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
    --tasks pope \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix pope \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=7 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
    --tasks llava_in_the_wild \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_in_the_wild \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=7 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix mme \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=7 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
    --tasks mmvet \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix mmvet \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=7 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
    --tasks scienceqa_img,gqa,mmbench_en,mmbench_cn,pope,llava_in_the_wild,mme,mmvet,seedbench,vizwiz_vqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix seedbench \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


python3 -m accelerate.commands.launch \
    --num_processes=7 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
    --tasks vizwiz_vqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix vizwiz_vqa \
    --output_path ./logs/ \
    --verbosity=DEBUG \
    --hf_hub_log_args hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 

