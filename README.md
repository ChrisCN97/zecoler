# Zecoler

The code implement of paper "Zero-Shot Program Representation Learning".

## Abstract

Learning program representations has been the core prerequisite of code intelligent tasks such as code search and code clone detection. The state-of-the-art pre-trained models such as CodeBERT require the availability of large-scale code corpora. However, gathering training samples can be costly and infeasible for domain-specific languages such as Solidity for smart contracts. In this paper, we propose Zecoler, a zero-shot learning approach for code representations. Zecoler is built upon a pre-trained programming language model. In order to elicit knowledge from the pre-trained models efficiently, Zecoler casts the downstream tasks to the same form of pre-training tasks by inserting trainable prompts into the original input. Then, it employs the prompt learning technique which optimizes the pre-trained model by merely adjusting the original input. This enables the representation model to efficiently fit the scarce task-oriented data while reusing pre-trained knowledge. We evaluate Zecoler in three code intelligent tasks in two program languages that have no training samples, namely, Solidity and Go, with model trained in corpora of common languages such as Java. Experimental results show that our approach significantly outperforms baseline models in both zero-shot and few-shot settings.

## Structure

* dataset: Preprocess source data with util.py for each downstream tasks. There is an example dataset in 
  clone_detection/Java.
* draw: Draw the picture in the paper.
* method: The source code of Zecoler and baselines. The data is processed again in the dataset folder. There 
  is an example in **/dataset/clone_detection/Java/500.
* run: The running shell of code can be constructed in finetune.py and ptuning.py.
