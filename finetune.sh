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
cd download_data
python preprocess_llava_onevision_parquet.py
cd ..

export HF_TOKEN=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
export HF_HOME=/root/autodl-tmp/huggingface
huggingface-cli login --token $HF_TOKEN

bash /root/autodl-tmp/lmms-eval/LLaVA-NeXT/scripts/train/finetune_ov_FigureQA_MathV360K.sh
