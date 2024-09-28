import sentencepiece as spm
import json 
import jieba
def create_train_corpus_psyqa():
    filepaths = ["./data/PsyQa/train.json", "./data/PsyQa/valid.json", "./data/PsyQa/test.json"]
    write_file = open("./tokenizer/target_unigram_model/spm_train_psyqa.txt", "w")
    for filepath in filepaths:
        with open(filepath, encoding="utf-8") as input_file:
            dataset = json.load(input_file)
            idx = 0 
            for meta_data in dataset:
                for ans in meta_data["answers"]:
                    # translator = str.maketrans(" \n", "\u2582\u2583")
                    # seg_list = list(ans["answer_text"])
                    # write_file.write(" ".join(seg_list)+ "\n")
                    write_file.write(ans["answer_text"]+ "\n")
                    
def train():
    spm.SentencePieceTrainer.train(model_prefix="./tokenizer/target_unigram_model_psyqa/target.model", input='./tokenizer/target_unigram_model_psyqa/spm_train_psyqa.txt', model_type="unigram", vocab_size=12000, split_by_whitespace=False, max_sentencepiece_length=32)
    
def test():
    sp = spm.SentencePieceProcessor(model_file='./test_model.model')
    # print(sp.sample_encode_and_score('在你的问题描述中，你能够感知到自己是一个有目标的人，但是你却缺乏【能量】，这能量更学术一点的来说，就是动机', num_samples=5, alpha=0.1, wor=True))
    # print(sp.encode('在你的问题描述中，你能够感知到自己是一个有目标的人，但是你却缺乏【能量】，这能量更学术一点的来说，就是动机', out_type=str))
    
# train()
create_train_corpus_psyqa()