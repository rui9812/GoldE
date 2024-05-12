# GoldE
Generalizing Knowledge Graph Embedding with Universal Orthogonal Parameterization

(This paper is accepted by ICML 2024)

## Requirements
- pytorch == 1.8.0
- numpy == 1.19.2
- scikit-learn == 0.23.2

## Data
 - *entities.dict*: a dictionary map entities to unique ids
 - *relations.dict*: a dictionary map relations to unique ids
 - *train.txt*: the KGE model is trained to fit this data set
 - *valid.txt*: create a blank file if no validation data is available
 - *test.txt*: the KGE model is evaluated on this data set

## Usage
All training commands are listed in [best_config.sh](./best_config.sh). 
For example, you can run the following commands to train GoldE on WN18RR datasets.
```
# WN18RR
bash run.sh GoldE wn18rr 0 0 0 1000 200 800 12 10 0.666435178264418 0.99 0.5 6.0 1.1 0.003 60000 20000 16 0.185933138885153 -sf
```

## Acknowledgement
We refer to the code of [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding) and [HousE](https://github.com/rui9812/HousE). Thanks for their contributions.
