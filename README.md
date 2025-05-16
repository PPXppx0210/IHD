# LHD: A LLM-Aided Heterogeneous Diffusion Model
A deep learning model for predicting information diffusion in social networks using LLM-aided interest-aware heterogeneous diffusion graphs.

## Features
- Integrates large language models (LLMs) for topic classification
- Constructs interest-aware heterogeneous diffusion graphs (IHDGs)

## Installation
```bash
pip install -r requirements.txt
```

## Usage

#### 1. Classifying tweets with Wikipedia
```bash
cd topicGPT
./run_tmux_sessions.sh
```

#### 2. Constructing IHDGs
```bash
cd preprocess-llm
python gt_ft_mapping.py
python IHDG_constructing.py
```

#### 3. Interest-aware User Representation Learning
```bash
cd src
nohup python train_series.py --model lhd --tensorboard-log lhd-tw-basic_1 --graph-filename full --window-size 7 --graph-topk 20 --dataset Twitter-Huangxin --batch-size 32 --hidden-units 16,16 --heads 4,4 --gpu cuda:8 &> lhd-tw-basic_1.txt &
```

## Acknowledgments

Our code is partly based on [topicGPT](https://github.com/chtmp223/topicGPT) repository. We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.
