# AAGCN-ACSA
EMNLP 2021

# Introduction
This repository was used in our paper:  
  
**Beta Distribution Guided Aspect-aware Graph for Aspect Category Sentiment Analysis with Affective Knowledge**
<br>
Bin Liang<sup>\*</sup>, Hang Su<sup>\*</sup>, Rongdi Yin, Lin Gui, Min Yang, Qin Zhao, Xiaoqi Yu, and Ruifeng Xu. *Proceedings of EMNLP 2021*
  
Please cite our paper and kindly give a star for this repository if you use this code.

## Requirements

* Python 3.6
* PyTorch 1.0.0
* SpaCy 2.0.18
* numpy 1.15.4

## Usage

* Install [SpaCy](https://spacy.io/) package and language models with
```bash
pip install spacy
```
and
```bash
python -m spacy download en
```
* Generate aspect-focused graph with
```bash
python generate_graph_for_aspect.py
```
* Generate inter-aspect graph with
```bash
python generate_position_con_graph.py
```

## Training
* Train with command, optional arguments could be found in [train.py](/train.py) \& [train_bert.py](/train_bert.py)


* Run intergcn: ```./run_intergcn.sh```

* Run afgcn: ```./run_afgcn.sh```



* Run intergcn_bert: ```./run_intergcn_bert.sh```

* Run afgcn_bert: ```./run_afgcn_bert.sh```



## Citation

The BibTex of the citation is as follow:

```bibtex
```
