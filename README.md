# Zecoler

The code implement of paper "Zero-Shot Code Representation Learning for Program Understanding and Generation".


## Abstract

Learning program representations has been the core prerequisite of
code understanding and generation tasks (e.g., code search and code summa-
rization). The state-of-the-art pre-trained models such as CodeBERT require the
availability of large-scale code corpora. However, gathering training samples can
be costly and infeasible for domain-specific languages such as Solidity for smart
contracts or the scenario where downstream tasks need to be solved in specific
project. In this paper, we propose Zecoler, a zero-shot learning approach for code
representations. Zecoler is built upon a pre-trained programming language model.
In order to elicit knowledge from the pre-trained models efficiently, Zecoler casts
the downstream tasks to the same form of pre-training tasks by inserting trainable
prompts into the original input. These prompts will guide PLM to generate better
output just like human tells machine what to do. Then, it employs the prompt
learning technique to optimize the pre-trained model and search for the most suit-
able continuous prompts automatically. This enables the representation model to
efficiently fit the downstream tasks through the dataset in source language domain
and then reuse pre-trained and continuous trained knowledge in PLM for target
language domain in zero-shot style. We evaluate Zecoler in three code understand-
ing and two generation tasks in multiple programming languages that have no
training samples, e.g., Solidity and Go, with model trained in corpora of common
languages such as Java. Experimental results show that our approach significantly
outperforms baseline models under both zero-shot and few-shot settings.

## Structure

* dataset: Preprocess source data with util.py for each downstream tasks. There is an example dataset in 
  clone_detection/Java.
* draw: Draw the picture in the paper.
* method: The source code of Zecoler and baselines. The data is processed again in the dataset folder. There 
  is an example in **/dataset/clone_detection/Java/500.
* run: The running shell of code can be constructed in finetune.py and ptuning.py.
* generate: the code for code generation tasks
