# one-click-deployment

```
mkdir project68
cd project68
mkdir dataset68
cd dataset68
pip install -U "huggingface_hub[cli]"
export HF_TOKEN=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
huggingface-cli login --token $HF_TOKEN

huggingface-cli download wei682/LLaVA_data pretrain/blip_laion_cc_sbu_558k.json  --local-dir . --repo-type dataset
huggingface-cli download wei682/LLaVA_data pretrain/images.tar.gz  --local-dir . --repo-type dataset
cd pretrain/
sudo apt install pigz
mkdir -p images && nice -n -20 ionice -c 2 -n 0 pv -B 512M images.tar.gz | tar -I "pigz -d -p $(nproc)" -x -C images

huggingface-cli download wei682/LLaVA_data finetune/llava_v1_5_mix665k.json  --local-dir . --repo-type dataset
huggingface-cli download wei682/LLaVA_data finetune/data/LLaVA-Pretrain.tar.gz  --local-dir . --repo-type dataset
huggingface-cli download wei682/LLaVA_data finetune/data/coco.tar.gz  --local-dir . --repo-type dataset
huggingface-cli download wei682/LLaVA_data finetune/data/coco2014_val_qa_eval.tar.gz  --local-dir . --repo-type dataset
huggingface-cli download wei682/LLaVA_data finetune/data/vg.tar.gz  --local-dir . --repo-type dataset
huggingface-cli download wei682/LLaVA_data finetune/data/textvqa.tar.gz  --local-dir . --repo-type dataset
huggingface-cli download wei682/LLaVA_data finetune/data/prompts.tar.gz  --local-dir . --repo-type dataset
huggingface-cli download wei682/LLaVA_data finetune/data/ocr_vqa.tar.gz  --local-dir . --repo-type dataset
huggingface-cli download wei682/LLaVA_data finetune/data/gqa.tar.gz  --local-dir . --repo-type dataset
huggingface-cli download wei682/LLaVA_data finetune/data/eval_part_aa  --local-dir . --repo-type dataset
huggingface-cli download wei682/LLaVA_data finetune/data/eval_part_ab  --local-dir . --repo-type dataset
huggingface-cli download wei682/LLaVA_data finetune/data/eval_part_ac  --local-dir . --repo-type dataset
huggingface-cli download wei682/LLaVA_data finetune/data/eval_part_ad  --local-dir . --repo-type dataset
```
