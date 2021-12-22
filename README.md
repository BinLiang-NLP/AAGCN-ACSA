# AAGCN-ACSA
**The code of this repository is constantly being updated...** 

**Please look forward to it!**

# Introduction
This repository was used in our paper: 
  
**Beta Distribution Guided Aspect-aware Graph for Aspect Category Sentiment Analysis with Affective Knowledge**
<br>
Bin Liang<sup>#</sup>, Hang Su<sup>#</sup>, Rongdi Yin, Lin Gui, Min Yang, Qin Zhao, Xiaoqi Yu, and Ruifeng Xu. *Proceedings of EMNLP 2021*
  
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
python3 -m spacy download en
```


## Training
* For non-BERT implementation. Please train with command, optional arguments could be found in [train.py](/train.py)
```bash
./train.sh
```
* For BERT-based implementation. Please train with command, optional arguments could be found in [train_bert.py](/train_bert.py)
```bash
./train_bert.sh
```

## See also
* The process of graph generation could be found at [./dataset/generate_graph.py](/dataset/generate_graph.py).
* The source of [SenticNet](https://sentic.net/) could be found at https://sentic.net/downloads/.
* The source of [ConceptNet] (https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14972/14051) could be downloaded ate https://github.com/commonsense/conceptnet5.


## Citation

The BibTex of the citation is as follow:

```bibtex
@inproceedings{liang-etal-2021-beta,
    title = "Beta Distribution Guided Aspect-aware Graph for Aspect Category Sentiment Analysis with Affective Knowledge",
    author = "Liang, Bin  and
      Su, Hang  and
      Yin, Rongdi  and
      Gui, Lin  and
      Yang, Min  and
      Zhao, Qin  and
      Yu, Xiaoqi  and
      Xu, Ruifeng",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.19",
    pages = "208--218",
    abstract = "In this paper, we investigate the Aspect Category Sentiment Analysis (ACSA) task from a novel perspective by exploring a Beta Distribution guided aspect-aware graph construction based on external knowledge. That is, we are no longer entangled about how to laboriously search the sentiment clues for coarse-grained aspects from the context, but how to preferably find the words highly related to the aspects in the context and determine their importance based on the public knowledge base. In this way, the contextual sentiment clues can be explicitly tracked in ACSA for the aspects in the light of these aspect-related words. To be specific, we first regard each aspect as a pivot to derive aspect-aware words that are highly related to the aspect from external affective commonsense knowledge. Then, we employ Beta Distribution to educe the aspect-aware weight, which reflects the importance to the aspect, for each aspect-aware word. Afterward, the aspect-aware words are served as the substitutes of the coarse-grained aspect to construct graphs for leveraging the aspect-related contextual sentiment dependencies in ACSA. Experiments on 6 benchmark datasets show that our approach significantly outperforms the state-of-the-art baseline methods.",
}
```
