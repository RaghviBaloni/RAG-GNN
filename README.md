# RAGE

This repository contains the source code for the implementation of "RAGE: RAG Enhanced LLM Eplainer for Heterogeneous Graphs"

## Environment Setup
```sh
conda create --name rage python=3.9 -y
conda activate rage


conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

pip install peft
pip install pandas
pip install ogb
pip install transformers
pip install wandb
pip install sentencepiece
pip install torch_geometric
pip install datasets
pip install pcst_fast
pip install gensim
pip install scipy==1.12
pip install protobuf
pip install openai
pip install langchain-community

```

## Data Preprocessing
Download the dataset from the links below:

Goodreads: https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html

DBLP: https://originalstatic.aminer.cn/misc/dblp.v12.7z

Use the preprocessing files under dataset folder to preprocess the datasets

```sh
cd dataset/
```

Navigate to the preprocessig folder under ``src/dataset/preprocess/`` and modify the path for the training data

```sh
python -m src.dataset.preprocess.goodreads
python -m src.dataset.preprocess.dblp
python -m src.dataset.goodreads
python -m src.dataset.dblp
```

## LLM Inference for generating Predictions and Explanations

Replace path to the llm in the ``src/model/__init__.py`` if needed

```sh
python inference.py --dataset name of the dataset --model_name inference_llm --llm_model_name Name of the llm (openai/claude)
python evaluate.py 
```





