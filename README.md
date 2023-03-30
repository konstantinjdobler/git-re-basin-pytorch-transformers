# Git Re-Basin in Pytorch for HuggingFace Transformers

PyTorch implementation of the weight matching algorithm from the [Git Re-Basin paper](https://arxiv.org/abs/2209.04836) by Ainsworth et al. Their implementation uses JAX (https://github.com/samuela/git-re-basin) and an adaptation to PyTorch has been published (https://github.com/themrzmaster/git-re-basin-pytorch).

At the time of writing I couldn't find an implementation for Huggingface `transformers` (or any transformer-style models), so I'm providing my implementation (based on the PyTorch adaptation) here. It's moderately tested on `bert-base-uncased`, `roberta-base`, and `xlm-roberta-base`.

### Basic Setup
`pip install -r requirements.txt` or `conda env create --file environment.yml`.

Then run `python weight_matching.py --model roberta-base`.

Have fun!