AGNews Classification Project on GPT Architecture: A Finetuning Experient on Minimind
=================================

This project studies the robustness and generalization of GPT-style models on the AGNews classification task under clean and noisy settings.


Directory Structure
---------------------------------

eval_results/
Stores all evaluation outputs  
- Includes accuracy, precision, recall, and F1-score  
- Includes per-sample prediction details for error analysis  

out_agnews/
Stores all trained model checkpoints  
- Includes full SFT and further SFT (noisy) models  
- The evaluation script loads models from this folder by default  

trainer/
Contains all training scripts  
- Clean SFT training  
- Further SFT on noisy data  

eval_agnews.py
Evaluation script for AGNews classification  
- Supports both clean and noisy valid/test evaluation  


Usage Instructions
---------------------------------

Please refer to the following section inside each script:

    HOW TO USE

All necessary launch commands for:
- evaluation
- clean SFT
- further noisy SFT

are provided directly inside the scripts.


Supported Datasets
---------------------------------

clean_train.jsonl  
clean_valid.jsonl  
clean_test.jsonl  

noisy_train.jsonl  
noisy_valid.jsonl  
noisy_test.jsonl  


Project Features
---------------------------------

- GPT-style model evaluation on AGNews
- Clean supervised fine-tuning (SFT)
- Noisy further supervised fine-tuning (further SFT)
- Systematic comparison between clean and noisy robustness
- Study of noise as data augmentation and regularization


Credits
---------------------------------

This project is based on the MiniMind framework developed by:

    https://github.com/jingyaogong/minimind

The pretrained checkpoint used in this project:

    pretrain_768.pth

is provided by the same author and obtained from:

    https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch/files
