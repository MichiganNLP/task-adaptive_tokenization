import sys
import os
sys.path.append("/home/lsiyang/scratch/variable-text-segmentation")
import data_utils.sentencepiece_model_pb2 as proto_model
m = proto_model.ModelProto()
name_or_path = "tokenizer/customed_LLaMA_ch_psyqa/tokenizer.model"
m.ParseFromString(open(name_or_path, "rb").read())    
pieces = m.pieces 
vocab = []  
print(len(pieces))
for id, piece in enumerate(pieces):
    vocab.append((piece.piece, piece.score)) 
print(len(vocab))
print(vocab[-1])
print(os.path.split(name_or_path)[0] +"/"+ os.path.split(name_or_path)[-1].split(".")[0] +".vocab")
with open(os.path.split(name_or_path)[0] +"/"+ os.path.split(name_or_path)[-1].split(".")[0] +".vocab", mode = "w", newline='') as f:
    write_text = ""
# with open(os.path.split(self.name_or_path)[0] +"/"+ "target_0.vocab", "w") as f:
    for idx, item in enumerate(vocab):
        write_text = write_text + item[0] + "\t" + str(round(item[1],5)) + "\n"       
    print(len(write_text.split("\n")))
    f.write(write_text)