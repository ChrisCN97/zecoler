# zeroshot

The code implement of paper "Zero-Shot Program Representation Learning".

## structure

* dataset: Preprocess source data with util.py for each downstream tasks. There is an example dataset in 
  clone_detection/Java.
* draw: Draw the picture in the paper.
* method: The source code of Zecoler and baselines. The data is processed again in the dataset folder. There 
  is an example in **/dataset/clone_detection/Java/500.
* run: The running shell of code can be constructed in finetune.py and ptuning.py.