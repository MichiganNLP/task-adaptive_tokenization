# Task Adaptive Tokenization
A code implementation for the EMNLP 2023 paper "Task-Adaptive Tokenization: Enhancing Long-Form Text Generation Efficacy in Mental Health and Beyond"


```bibtex
@inproceedings{liu-etal-2023-task,
    title = "Task-Adaptive Tokenization: Enhancing Long-Form Text Generation Efficacy in Mental Health and Beyond",
    author = "Liu, Siyang  and
      Deng, Naihao  and
      Sabour, Sahand  and
      Jia, Yilin  and
      Huang, Minlie  and
      Mihalcea, Rada",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.944",
    doi = "10.18653/v1/2023.emnlp-main.944",
    pages = "15264--15281",
}
```



# Create a Task-adaptive Tokenizer

**1. training a sentencepiece vocabulary using your downstream corpus**

  see example in *./vocab_files/target_unigram_model_for_psyqa/train_spm_model.py*

  If you want to build a specialized vocabulary for other datasets, please see: *./vocab_files/target_unigram_model_for_psyqa/vocabulary_build.py*

**2. save the base vocabulary into a folder**

  create a directory under *./vocab_files*, and put all vocab files and config files under *./vocab_files/{dir}*. See an example in *./vocab_files/merged_vocab_from_llama_base_for_psyqa*

**3. run Build_TAT_from_BaseTokenizer in create_task_adptive_tokenizer_from_base.py**
  
  this script will build a task-adaptive tokenizer and save the newly merged vocab file into the output

# Project Reproducing
## download data
./data/PsyQa/loading_script.py will automatically prepare dataset we need. You usually just need this script.


## setting up the environment
You may need two environments to run:
open and follow the following command in ./install.sh

## training models

see ./train.sh, and change some parameters accordingly

## generating

see ./generate.sh, and change some parameters accordingly

