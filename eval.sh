git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
pip install -e .

git clone https://github.com/noahwei682/LLaVA-NeXT.git
cd LLaVA-NeXT

conda create -n llava-next python=3.10 -y
source activate llava-next
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
pip install flash-attn==2.5.2 --no-build-isolation

export HF_TOKEN=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
huggingface-cli login --token $HF_TOKEN

