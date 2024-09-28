# task-adaptive_tokenization
A code implementation for the EMNLP 2023 paper "Enhancing Long-form Text Generation Efficacy with Task-adaptive Tokenization"

# Create a Task-adaptive Tokenizer

1. training a sentencepiece vocabulary using your downstream corpus

see example in ./vocab_files/target_unigram_model_for_psyqa/train_spm_model.py

If you want to build a specialized vocabulary for other datasets, please see: ./vocab_files/target_unigram_model_for_psyqa/vocabulary_build.py

2. save the base vocabulary into a folder

create a directory under ./vocab_files, and put all vocab files and config files under ./vocab_files/{dir}. See example in ./vocab_files/merged_vocab_from_llama_base_for_psyqa/

3. run Build_TAT_from_BaseTokenizer in variable-text-segmentation/model/create_task_adptive_tokenizer_from_base.py
this script will build a task-adaptive tokenizer and save the newly merged vocab file into the output

# task-adaptive tokenization
## download data

./data/PsyQa/loading_script.py will automatically prepare dataset we need. You usually just need this script.


## setup the environment
You may need two environments to run:
open and follow the following command in ./install.sh

## training models

see ./train.sh, and change some parameters accordingly

## generating

see ./generate.sh, and change some parameters accordingly


## building a merged vocabulary from a base vocabulary and a specialized vocabulary
### training a specialized vocabulary
see ./tokenizer/target_unigram_model/train_spm_model.py. Just for psyqa data, the specialized vocabulary is already in the directory.
If you want to build a specialized vocabulary for other datasets, please see: ./tokenizer/target_unigram_model/vocabulary_build.py
### creating a drectory for a base vocabulary
create a directory under ./tokenizer/, and put all vocab files and config files under ./tokenizer/{dir}
### merging two vocabularies 
see the implementation of  test_ReindexAccordingWordpiece and test_BuildMappingFileTestCase in ./model/unit_test_customed_gpt2.py. Modify it accordingly.
