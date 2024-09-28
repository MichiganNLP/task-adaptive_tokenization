#!/bin/bash
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02-00:00:00
#SBATCH --mem-per-gpu=48g
#SBATCH --output=/home/%u/scratch/task_adaptive_tokenization/%x-%j.log
#SBATCH --job-name=training
#SBATCH --account=
# set up job
# cd ~/task_adaptive_tokenization/
# export PATH=~/miniconda/envs/mix/bin:$PATH
# conda init bash
# source ~/.bashrc
# conda activate mix
# export CUBLAS_WORKSPACE_CONFIG=:16:8

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES="1"
#把模型地址改到自己所训练好的模型路径上

vocab_size=10000

python3 train.py \
--model_name_or_path=./checkpoint/GPT2-961m \
--config_name=./checkpoint/PsyqaCustomed-GPT2-961m/config_${vocab_size}_mapping.json \
--use_bi_tokenizer=True \
--tokenizer_name=./tokenizer/cluecorpussmall \
--dataset_script=./data/PsyQa/loading_script.py \
--use_fast_tokenizer=False \
--model_type="customed" \
--block_size=1024 \
--per_device_train_batch_size=8 \
--num_train_epochs=30 \
--logging_steps=40000 \
--save_strategy="steps" \
--save_steps=40000 \
--eval_steps=40000 \
--evaluation_strategy="steps" \
--do_eval \
--output_dir=./output/merged2merged_psyqa_gpt_${vocab_size}_mapping/ \
--do_train \
--overwrite_output_dir \
--target_tokenizer_name=./tokenizer/customed-wordpiece/target_${vocab_size}.model \
--source_tokenizer_type="sentencepiece" \
--overwrite_cache \
--dataset_config_name="wo strategy" \
--weight_decay=0.1 \
--learning_rate=1e-5 \
--adam_epsilon=1e-6 \
--adam_beta1=0.9 \
--adam_beta2=0.999 \
--max_grad_norm=1.0 \
--warmup_ratio=0.1
# --max_steps=1000 \






