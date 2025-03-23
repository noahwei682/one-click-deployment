conda create -n llava-next python=3.10 -y
pip install --upgrade pip  # Enable PEP 660 support.

git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
pip install -e .

git clone https://github.com/noahwei682/llava-ov-ewc-ms.git
cd llava-ov-ewc-ms/

pip install -e ".[train]"
pip install flash-attn==2.5.2 --no-build-isolation
pip install ipdb


pip install huggingface_hub
export HF_TOKEN=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
huggingface-cli login --token $HF_TOKEN
# local-dir=/home/aiscuser

mkdir msdata
huggingface-cli download wei682/amazon images_20250321.tar.gz --repo-type dataset  --local-dir ./msdata
cd msdata
mkdir images
tar -xvzf images_20250321.tar.gz -C ./images

huggingface-cli download wei682/amazon meta_Electronics_exist_gene_ITdata.json --repo-type dataset  --local-dir .
huggingface-cli download wei682/amazon meta_Arts_Crafts_and_Sewing_exist_gene_ITdata.json --repo-type dataset  --local-dir .
huggingface-cli download wei682/amazon meta_Industrial_and_Scientific_exist_gene_ITdata.json --repo-type dataset  --local-dir .
huggingface-cli download wei682/amazon meta_Health_and_Household_exist_gene_ITdata.json --repo-type dataset  --local-dir .
huggingface-cli download wei682/amazon meta_Movies_and_TV_exist_gene_ITdata.json --repo-type dataset  --local-dir .
huggingface-cli download wei682/amazon meta_Beauty_and_Personal_Care_exist_gene_ITdata.json --repo-type dataset  --local-dir .
huggingface-cli download wei682/amazon meta_Electronics_exist_gene_ITdata.json --repo-type dataset  --local-dir .
huggingface-cli download wei682/amazon meta_Appliances_exist_gene_ITdata.json --repo-type dataset  --local-dir .
huggingface-cli download wei682/amazon meta_Cell_Phones_and_Accessories_exist_gene_ITdata.json --repo-type dataset .
huggingface-cli download wei682/amazon meta_Grocery_and_Gourmet_Food_exist_gene_ITdata.json --repo-type dataset  --local-dir .
huggingface-cli download wei682/amazon meta_Sports_and_Outdoors_exist_gene_ITdata.json --repo-type dataset  --local-dir .
huggingface-cli download wei682/amazon meta_Magazine_Subscriptions_exist_gene_ITdata.json --repo-type dataset  --local-dir .
huggingface-cli download wei682/amazon meta_Subscription_Boxes_exist_gene_ITdata.json --repo-type dataset  --local-dir .
huggingface-cli download wei682/amazon meta_Toys_and_Games_exist_gene_ITdata.json --repo-type dataset  --local-dir .
huggingface-cli download wei682/amazon meta_All_Beauty_exist_gene_ITdata.json --repo-type dataset  --local-dir .
huggingface-cli download wei682/amazon meta_Patio_Lawn_and_Garden_exist_gene_ITdata.json --repo-type dataset  --local-dir .
huggingface-cli download wei682/amazon meta_Gift_Cards_exist_gene_ITdata.json --repo-type dataset  --local-dir .
