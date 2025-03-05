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
vmtouch -t images.tar.gz &>/dev/null; mkdir -p images && TMPDIR=/dev/shm pv -B 1G -i 0.5 -W images.tar.gz | tar --warning=no-timestamp --checkpoint=.5000 -I "pigz -d -p $(nproc) --fast --rsyncable" -x -C images --checkpoint-action=ttyout='.'


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
cd finetune/
cd data/
vmtouch -t coco.tar.gz &>/dev/null; mkdir -p coco && TMPDIR=/dev/shm pv -B 1G -i 0.5 -W coco.tar.gz | tar --warning=no-timestamp --checkpoint=.5000 -I "pigz -d -p $(nproc) --fast --rsyncable" -x -C coco --checkpoint-action=ttyout='.'
vmtouch -t coco2014_val_qa_eval.tar.gz &>/dev/null; mkdir -p coco2014_val_qa_eval && TMPDIR=/dev/shm pv -B 1G -i 0.5 -W coco2014_val_qa_eval.tar.gz | tar --warning=no-timestamp --checkpoint=.5000 -I "pigz -d -p $(nproc) --fast --rsyncable" -x -C coco2014_val_qa_eval --checkpoint-action=ttyout='.'
vmtouch -t vg.tar.gz &>/dev/null; mkdir -p vg && TMPDIR=/dev/shm pv -B 1G -i 0.5 -W vg.tar.gz | tar --warning=no-timestamp --checkpoint=.5000 -I "pigz -d -p $(nproc) --fast --rsyncable" -x -C vg --checkpoint-action=ttyout='.'
vmtouch -t textvqa.tar.gz &>/dev/null; mkdir -p textvqa && TMPDIR=/dev/shm pv -B 1G -i 0.5 -W textvqa.tar.gz | tar --warning=no-timestamp --checkpoint=.5000 -I "pigz -d -p $(nproc) --fast --rsyncable" -x -C textvqa --checkpoint-action=ttyout='.'
vmtouch -t prompts.tar.gz &>/dev/null; mkdir -p prompts && TMPDIR=/dev/shm pv -B 1G -i 0.5 -W prompts.tar.gz | tar --warning=no-timestamp --checkpoint=.5000 -I "pigz -d -p $(nproc) --fast --rsyncable" -x -C prompts --checkpoint-action=ttyout='.'
vmtouch -t ocr_vqa.tar.gz &>/dev/null; mkdir -p ocr_vqa && TMPDIR=/dev/shm pv -B 1G -i 0.5 -W ocr_vqa.tar.gz | tar --warning=no-timestamp --checkpoint=.5000 -I "pigz -d -p $(nproc) --fast --rsyncable" -x -C ocr_vqa --checkpoint-action=ttyout='.'
vmtouch -t gqa.tar.gz &>/dev/null; mkdir -p gqa && TMPDIR=/dev/shm pv -B 1G -i 0.5 -W gqa.tar.gz | tar --warning=no-timestamp --checkpoint=.5000 -I "pigz -d -p $(nproc) --fast --rsyncable" -x -C gqa --checkpoint-action=ttyout='.'
```
