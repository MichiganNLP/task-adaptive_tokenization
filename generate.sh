#!/bin/bash
#SBATCH --partition=spgpu,gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01-12:00:00
#SBATCH --mem-per-gpu=40g
#SBATCH --output=/home/%u/scratch/task_adaptive_tokenization/%x-%j.log
#SBATCH --job-name=generating
#SBATCH --cpus-per-gpu=2
#SBATCH --account=
# set up job
cd ~/task_adaptive_tokenization/
export PATH=~/miniconda/envs/mix/bin:$PATH
conda init bash
source ~/.bashrc
conda activate mix

# export CUDA_VISIBLE_DEVICES="0"

export CUDA_LAUNCH_BLOCKING=1
#把模型地址改到自己所训练好的模型路径上
vocab_size=10000
python3 train.py \
--model_name_or_path=./output/merged2merged_psyqa_gpt_${vocab_size}_mapping/checkpoint-160000 \
--config_name=./output/merged2merged_psyqa_gpt_${vocab_size}_mapping/checkpoint-160000 \
--use_bi_tokenizer=True \
--tokenizer_name=./output/merged2merged_psyqa_gpt_${vocab_size}_mapping/checkpoint-160000 \
--dataset_script=./data/PsyQa/loading_script.py \
--use_fast_tokenizer=False \
--model_type="customed" \
--block_size=1024 \
--do_generation_test \
--output_dir=./output/merged2merged_psyqa_gpt_${vocab_size}_mapping/ \
--overwrite_output_dir \
--target_tokenizer_name=./tokenizer/customed-wordpiece/target_${vocab_size}.model \
--source_tokenizer_type="sentencepiece" \
--overwrite_cache \
--dataset_config_name="wo strategy" \
--generate_length=1024 \
--k=50 \
--p=1 \
--temperature=1 \
--repetition_penalty=2 \
--num_beams=3 \
--renormalize_logits=True \
--exponential_decay_length_penalty="(0,0.997)" \



