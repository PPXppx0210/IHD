# IHD: An Interest-aware Heterogeneous Diffusion Model
A deep learning model for predicting information diffusion in social networks using interest-aware heterogeneous diffusion graphs (IHDGs).

## Features
- Integrates large language models (LLMs) for topic classification
- Constructs interest-aware heterogeneous diffusion graphs (IHDGs)

## Installation
```bash
conda env create -f environment.yml
conda activate IHD
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
python preprocess_twitter.py --param 6
python preprocess_weibo.py --param 6
```

#### 3. Interest-aware User Representation Learning
```bash
cd src
python train_series.py --model heteredgegat --tensorboard-log IHD-tw-basic_1 --graph-filename full --window-size 7 --tw-version 6 --dataset Twitter-Huangxin --batch-size 32 --hidden-units 16,16 --heads 4,4 --gpu cuda:7
```

## Acknowledgments

Our code is partly based on [topicGPT](https://github.com/chtmp223/topicGPT) repository. We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.
