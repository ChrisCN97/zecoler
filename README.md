# Zecoler

The code implement of paper "Zero-Shot Code Representation Learning via Prompt Tuning".


## Abstract

Learning code representations has been the core prerequisite of of many software engineering tasks such as code clone detection and code generation. The state-of-the-art pre-trained language models (PLMs) such as CodeBERT require an amount of downstream data for fine-tuning. However, gathering training sam- ples can be prohibitively expensive and impractical for domain-specific languages or project-specific tasks. Besides, pre-training and downstream tasks are usually heterogeneous, which make it hard to fully explore the knowledge learned during pre-training. In this paper, we propose Zecoler, a zero-shot approach for learning code representations. Zecoler is built upon a pre-trained programming language model. In order to elicit knowledge from the PLMs efficiently, Zecoler casts the downstream tasks to the same form of pre-training objectives by inserting train- able prompts into the original input. These prompts can guide PLMs on how to generate better results. Subsequently, we employ the prompt tuning technique to search for the optimal prompts for PLMs automatically. This enables the repre- sentation model to efficiently fit the downstream tasks through fine-tuning on the dataset in source language domain and then reuse the pre-trained knowledge for the target domain in a zero-shot style. We evaluate Zecoler in five code intelli- gence tasks including code clone detection, code search, method name prediction, code summarization, and code generation. We experiment in multiple program- ming languages without giving labeled samples, e.g., Solidity and Go, with model trained in corpora of common languages such as Java. The results show that our approach significantly outperforms baseline models under the zero-shot setting. For example, the accuracy of code search is improved by 30% compared to fine- tuning. In addition, qualitative analysis demonstrates its superior generalizability under both cross-lingual and monolingual few-shot settings.

## Structure

* dataset: Preprocess source data with util.py for each downstream tasks. There is an example dataset in 
  clone_detection/Java.
* draw: Draw the picture in the paper.
* method: The source code of Zecoler and baselines. The data is processed again in the dataset folder. There 
  is an example in **/dataset/clone_detection/Java/500.
* run: The running shell of code can be constructed in finetune.py and ptuning.py.
* generate: the code for code generation tasks
