# LEONA
Linguistically-Enriched and Context-Aware Cross-DomainZero-shot Slot Filling

### 0. Download data from the respective websites that host the public datasets: 
i. SNIPS: already copied in the dir (rawdata/snips)
ii. ATIS: already copied in the dir (rawdata/atis)
iii. MultiWOZ2.2: https://github.com/budzianowski/multiwoz/tree/master/data (not copied because of its size)
iv. SGD: https://github.com/google-research-datasets/dstc8-schema-guided-dialogue (not copied because of its size)


### 1. Preprocessing the datasets

i. Put data data into respective domains
run: preprocessing/extract_all_utterances.py -dataset (one value from  snips, atis, multiwoz, sgd)

ii. Generate data for experiments
run: preprocessing/generate_data_for_experiments.py -dataset (one value from  snips, atis, multiwoz, sgd)


### 2. Train the E2E model for ZS slot filling
run: model/end2end_model.py -dataset (one value from  snips, atis, multiwoz, sgd)


#### Dependencies:
1. Spacy >= 2.2.1
2. Pytorch >=1.3.0
3. TorchText >= 0.4.0
4. AllenNLP ==1.0.0
5. sklearn >=0.19
6. Numpy >= 1.17.3
7. python >=3.7.4