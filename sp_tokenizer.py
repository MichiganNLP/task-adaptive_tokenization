# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for OpenAI GPT."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import itertools

import json
from io import open
import sentencepiece as spm
import jieba
import math
import data_utils.sentencepiece_model_pb2 as proto_model
import os
import copy
from typing import Optional, Dict, Union, List, Tuple
from collections.abc import Mapping, Sized
from transformers.utils import PaddingStrategy, TensorType, logging
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, TruncationStrategy, BatchEncoding
try:
    from functools import lru_cache
except ImportError:
    # Just a dummy decorator to get the checks to run on python2
    # because honestly I don't want to support a byte-level unicode BPE tokenizer on python 2 right now.
    def lru_cache():
        return lambda func: func
import torch
logger = logging.get_logger(__name__)
    
# Define type aliases and NamedTuples
TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]
VERY_LARGE_INTEGER = int(1e30)  # This is used to set the max input length for a model with infinite size input
LARGE_INTEGER = int(1e20)  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER



class BiTokenizer(object):
    def __init__(self, tokenizer0, tokenizer1 = None):
        self.source_tokenizer = tokenizer0
        if tokenizer1 is not None:
            self.target_tokenizer = tokenizer1
        else:
            self.target_tokenizer = self.source_tokenizer

class SPTokenizer(PreTrainedTokenizerBase):

    def __init__(self, **kwargs):
        
        # self.max_len = max_len if max_len is not None else int(1e12)
        # self.encoder = json.load(open(vocab_file))
        # self.decoder = {v:k for k,v in self.encoder.items()}
        if "name_or_path" not in kwargs:
            raise ValueError("name_or_path is required")
        self.name_or_path = kwargs["name_or_path"]
        self.special_tokens_map_file = kwargs.pop("special_tokens_map_file", os.path.dirname(self.name_or_path)+"/special_tokens_map.json")
        if os.path.exists(os.path.dirname(self.name_or_path)+"/special_tokens_map.json"):
            special_tokens = json.load(open(self.special_tokens_map_file))
            print("special_tokens:", special_tokens)
            kwargs.update(special_tokens)
        super().__init__( **kwargs)
        self.sp = spm.SentencePieceProcessor(model_file=self.name_or_path)
        if "align_pos" not in kwargs:
            raise ValueError("align_pos is required")
        print(kwargs["align_pos"])
        self.norm_align_pos(kwargs.pop("align_pos", 0))
 
        
    def norm_align_pos(self, align_pos: int):       
        self.align_pos = align_pos - self.sp.vocab_size() if align_pos >= 0 else align_pos
        print("align_pos:", self.align_pos, align_pos, self.sp.vocab_size())
        return self.align_pos
            
            
    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

    def __len__(self):
        return self.sp.get_piece_size()

    def tokenize(self, text, **kwargs):
        # """ Tokenize a string. """
     
        if 'tokenize_with_sampling' in kwargs:
            if kwargs['tokenize_with_sampling']:
                return self.sp.encode(text, nbest_size=-1, enable_sampling=kwargs['tokenize_with_sampling'], alpha=0.5, out_type=str)
            else:
                return self.sp.encode(text,  enable_sampling=kwargs['tokenize_with_sampling'], alpha=0, out_type=str)
        else:
            return self.sp.encode(text, enable_sampling=False, alpha=0, out_type=str)
            
    
    def _is_chinese_chars(self, cps):
        """Checks whether CP is the codepoint of a CJK character."""
        for item in cps:
            cp =ord(item)
            if not (
                (cp >= 0x4E00 and cp <= 0x9FFF)
                or (cp >= 0x3400 and cp <= 0x4DBF)  #
                or (cp >= 0x20000 and cp <= 0x2A6DF)  #
                or (cp >= 0x2A700 and cp <= 0x2B73F)  #
                or (cp >= 0x2B740 and cp <= 0x2B81F)  #
                or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
                or (cp >= 0xF900 and cp <= 0xFAFF)
                or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
            ):  #
                return False
        return True

    def encode(self, text, **kwargs):
        tokens = self.tokenize(text)    
        return self.sp.piece_to_id(tokens)
    
    def _convert_token_to_id(self, tokens, **kwargs):
        return self.sp.piece_to_id(tokens)
    
    def _convert_id_to_token(self, ids, **kwargs):
        return self.sp.id_to_piece(ids)

    def decode(self, tokens, **kwargs):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy().tolist()
        text = self.sp.decode(tokens)
        text = text.replace('\u2582', ' ').replace('\u2583', '\n')
        # text = text.replace(' ', '').replace('\u2582', ' ').replace('\u2583', '\n')
        return text
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]], **kwargs) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id(token))
        return ids

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False, **kwargs
    ) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            tokens.append(self._convert_id_to_token(index))
        return tokens
    

                     
        
    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = False,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens, **kwargs)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                ids = list(
                        itertools.chain(*(self.encode(t,  **kwargs) for t in text))
                    )
                return ids
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                        f"Input {text} is not valid. Should be a string, a list/tuple of strings or a list/tuple of"
                        " integers."
                )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None

        return self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )
        
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        <Tip>

        This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
        this inside your training loop.

        </Tip>

        Args:
            pair (`bool`, *optional*, defaults to `False`):
                Whether the number of added tokens should be computed in the case of a sequence pair or a single
                sequence.

        Returns:
            `int`: Number of special tokens added to sequences.
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = False,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens, **kwargs)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                ids = list(
                        itertools.chain(*(self.encode(t,  **kwargs) for t in text))
                    )
                return ids
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                        f"Input {text} is not valid. Should be a string, a list/tuple of strings or a list/tuple of"
                        " integers."
                    )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            else:
                ids, pair_ids = ids_or_pair_ids

            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))

        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        return BatchEncoding(batch_outputs)

    def _batch_prepare_for_model(
        self,
        batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
        """

        batch_outputs = {}
        for first_ids, second_ids in batch_ids_pairs:
            outputs = self.prepare_for_model(
                first_ids,
                second_ids,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        return batch_outputs
    
    def get_added_vocab(self):
        return {}
        

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        name_or_path = save_directory + "/" +  os.path.split(self.name_or_path)[-1]
        with open(name_or_path, 'wb') as f:
            f.write(open(self.name_or_path, 'rb').read())
                        
        with open(os.path.split(name_or_path)[0] +"/"+ os.path.split(self.name_or_path)[-1].split(".")[0] +".vocab", "w") as f:
            f.write(open(os.path.split(self.name_or_path)[0] +"/"+ os.path.split(self.name_or_path)[-1].split(".")[0] +".vocab", "r").read())
        
        special_tokens_path = save_directory + "/" + os.path.split(self.special_tokens_map_file)[-1]
        json.dump(json.load(open(self.special_tokens_map_file, "r")), open(special_tokens_path, "w"))
        
        return (name_or_path, special_tokens_path)
    
    def _add_tokens(self, new_tokens: List[str], special_tokens: bool = False) -> int:
        self.add_user_defined_tokens(new_tokens)   
        if special_tokens:
            if os.path.exists(self.special_tokens_map_file):
                sp_dict = json.load(open(self.special_tokens_map_file))
            else:
                sp_dict = {}  
            if "additional_special_tokens" in sp_dict:
                cur = sp_dict["additional_special_tokens"]
                for new_token in new_tokens:
                    if new_token not in cur:
                        cur.append(new_token)
                sp_dict["additional_special_tokens"] = cur
                print("sp_dict['additional_special_tokens']:",sp_dict["additional_special_tokens"])
            else:
                sp_dict["additional_special_tokens"] = new_tokens
            print("sp_dict:", sp_dict)
            json.dump(sp_dict, open(self.special_tokens_map_file, "w"))
        
        return len(new_tokens)

    def add_user_defined_tokens(self, added_tokens: List[str], type=4) -> None:
        tokens = added_tokens
        print(tokens, self.sp.piece_to_id(added_tokens))
        m = proto_model.ModelProto()
        m.ParseFromString(open(self.name_or_path, "rb").read())
        if self.sp.unk_id() not in self.sp.piece_to_id(added_tokens):
            logger.info("All user defined tokens to add have already been added. Nothing to change.")
            return    
        for token in tokens:
            if self.sp.is_unknown(self.sp.piece_to_id(token)):
                new_token = proto_model.ModelProto().SentencePiece()
                new_token.piece = token
                new_token.score = -100
                new_token.type = type # user_defined_tokens type 4
                m.pieces.insert(self.align_pos, new_token)
            else:
                raise ValueError("The token %s is already defined." % token)
        with open(self.name_or_path, 'wb') as f:
            f.write(m.SerializeToString())      
            
        with open(os.path.split(self.name_or_path)[0] +"/"+ os.path.split(self.name_or_path)[-1].split(".")[0] +".vocab", "w") as f:
            for item in m.pieces:
                f.write(item.piece + "\t" + str(round(item.score,5)) + "\n")
        self.sp = spm.SentencePieceProcessor(model_file=self.name_or_path) 
        

    def build_mapping_file(self, mapping_tokenizer: PreTrainedTokenizerBase, mapping_file: str, max_length: int=32, is_chinese_vocab=False):
        mapping_index = []
        for idx in range(self.align_pos, 0):
            mapped_text = self.sp.id_to_piece(self.sp.get_piece_size() + idx)
            if is_chinese_vocab==False:
                mapped_text = mapped_text.replace("▁", " ")
            if idx == self.align_pos:
                print("first mapped token:", mapped_text," pos:", self.align_pos)
            if self.sp.unk_id() == self.sp.get_piece_size() + idx:
                mapping_ids = [self.sp.get_piece_size() + idx]
            else:
                mapping_ids= mapping_tokenizer.encode(mapped_text,add_special_tokens=False)
                mapping_ids = [id for id in mapping_ids if id != mapping_tokenizer.unk_token_id]
            mapping_ids = mapping_ids[ : min(max_length, len(mapping_ids))]
            mapping_ids = mapping_ids + [self.pad_token_id] * (max_length - len(mapping_ids))
            mapping_index.append(mapping_ids)      
        json.dump(mapping_index, open(mapping_file, "w"))
        return mapping_index
            
        
                       
            
    def reindex_with_base_vocab(self, base_vocab_file, output_dir, control_tokens: Optional[Dict]=None, unknown_token: Optional[Dict]=None, byte_token: Optional[Dict]=None, downstream_seg_type: Optional[str]="wordpiece", is_chinese_vocab = True, whitespace_placeholder = None, base_score = -50, prefix_subword = "##"):
        '''
        This function is to append tokens trained from unigram model to a original vocabulary without 
        destroying the tokens' order of the original vocabulary.
        
        - base_vocab_file: the path of base vocab that you want to merge with a base vocab
        - output_dir: the path where you save the newly merged vocab file
        - control_tokens: to specify the special tokens; sentencepiece needs this information for correct tokenization
        - unknown_token: to specify the unknown token; sentencepiece needs this information for correct tokenization
        - byte_token: to specify byte_token; sentencepiece needs this information for correct tokenization
        - base_seg_type: the segmentation method/tool that the base vocab is created by; two options: wordpice, sentencepiece
        - is_chinese_vocab: whether the base vocab is Chinese
        - whitespace_placeholder: how the whitespace is presented in base vocab; check your base vocab to know it.
        - base_score: the default score for the token that does not receive a score from downstream vocab
        - prefix_subword: specify the prefix if you want to give a prefix to those subwords that must follow behind another word/subword in vocab. For english vocab, whitespace is used between independent words so it is neccesary to spefic the subwords that cannot be independent. As Chinese language does not have boundry between words, it is not required. 

        '''
        
        m = proto_model.ModelProto()
        m.ParseFromString(open(self.name_or_path, "rb").read())
        
        ext = base_vocab_file.split(".")[-1]
        if ext == "txt":
            according_vocab = open(base_vocab_file,"r").read().strip().split("\n")
            according_vocab = {i:according_vocab[i] for i in range(len(according_vocab))}
        elif ext == "json":
            according_vocab = json.load(open(base_vocab_file))
            according_vocab = {int(according_vocab[i]):i.replace(whitespace_placeholder, "▁") if whitespace_placeholder is not None else i for i in according_vocab  }
        elif ext == "vocab":
            according_vocab = open(base_vocab_file,"r").readlines()
            according_vocab = {i:according_vocab[i].split("\t")[0].strip() for i in range(len(according_vocab))}
            if downstream_seg_type == "sentencepiece":
                temp_vocab = {v:k for k,v in according_vocab.items()}
                for key, value in temp_vocab.items():
                    if len(key)>1 and "▁" == key[0]:
                        if key[1:] in temp_vocab:
                            according_vocab[temp_vocab[key[1:]]] = prefix_subword + key[1:]
                        according_vocab[value] = key[1:]
                del temp_vocab
        elif ext == "model"  and downstream_seg_type == "sentencepiece":
            m_according = proto_model.ModelProto()
            m_according.ParseFromString(open(base_vocab_file, "rb").read())    
            pieces = m_according.pieces 
            m.normalizer_spec.precompiled_charsmap = m_according.normalizer_spec.precompiled_charsmap
            m.trainer_spec.byte_fallback = m_according.trainer_spec.byte_fallback
            vocab = []  
            according_vocab = {}
            for id, piece in enumerate(pieces):
                vocab.append((piece.piece, piece.score)) 
                according_vocab[id] = piece.piece 
        else:
            raise Exception("No available vocab file to load")
            
                
            with open(os.path.split(base_vocab_file)[0] +"/"+ os.path.split(base_vocab_file)[-1].split(".")[0] +".vocab", "w") as f:
                for item in vocab:
                    f.write(item[0] + "\t" + str(round(item[1],5)) + "\n")          

        incremental_pieces = m.pieces
        new_pieces = []
        memory_dict = {}
        new_vocab = []
        has_unk = 0
        # write old vocabulary in its original order
        for id, word in according_vocab.items():
            new_token = proto_model.ModelProto().SentencePiece()
            # control_tokens ={"sep_token": "<|sep|>", "pad_token": "<|pad|>", "cls_token": "<|cls|>", "mask_token": "<|mask|>"}
            # unknown_token = {"unk_token": "<|unk|>"}
            if control_tokens and  word in list(control_tokens.values()):
                # ["[PAD]", "[CLS]", "[SEP]", "[MASK]",  "<s>", "</s>", "<pad>", "<mask>", "<cls>", "<sep>", "<eod>", "▃"]:
                # Don't know what "▃" is for but it's in special chars for BPE
                new_token.type = 4 #"USER Defined Token"
                new_token.score = 0
            elif "[unused" in word:
                new_token.type = 5 #"UNUSED"
                new_token.score = 0
            elif unknown_token and word in list(unknown_token.values()):
                # ["[UNK]", "<unk>"]:
                new_token.type = 2  #"UNKNOWN"
                new_token.score = -100
                has_unk = 1
            elif byte_token is not None and word in list(byte_token.values()):
                new_token.type = 6 #"BYTE"
                new_token.score = 0
            else:
                new_token.type = 1  #"NORMAL"  
                
            if new_token.type == 1:
                new_token.piece = word
                new_token.score = base_score *(len(new_token.piece)+1)/len(new_token.piece)  # theoretically this should be a value slightly higher than the biggest score shown among the incremental pieces                 
            else:
                new_token.piece = word
            memory_dict[new_token.piece] = id
            new_vocab.append((new_token.piece, new_token.score))
            new_pieces.append(new_token)
        according_vocab_length = max(list(memory_dict.values())) + 1
        
        
        for id, piece in enumerate(incremental_pieces):
            if "～～" == piece.piece:
                print(piece.piece in memory_dict, "～～") 
            #Assume all special tokens follow the original vocab, so we give up all special tokens defined in incremental vocab, ["UNKNOWN", "CONTROL", "UNUSED"]
            if piece.type != 1 and (has_unk != 0 or piece.type != 2): 
                continue
            # if the token is duplicated, overwrite the score to the newest one
            
            if piece.piece in memory_dict:
                # if (self._is_chinese_chars(piece.piece)  and len(piece.piece)==1 ):
                #     # print(id, piece.piece)
                if (self._is_chinese_chars(piece.piece)  ) or (not is_chinese_vocab):
                    new_pieces[memory_dict[piece.piece]].score = piece.score
                    new_vocab[memory_dict[piece.piece]] = (piece.piece, piece.score)                                    
            else:
                # if assign a specific unk_token by passing arguments, substitute existing unk_token with new one.
                if piece.type == 2 and unknown_token and piece.piece!=unknown_token:
                    piece.piece = unknown_token['unk_token']
                new_pieces.append(copy.deepcopy(piece))
                new_vocab.append((piece.piece, piece.score)) 
        
 
        del m.pieces[:]
        m.pieces.extend(new_pieces)
        self.name_or_path = output_dir + "/" +  os.path.split(self.name_or_path)[1]
        # self.name_or_path = os.path.split(self.name_or_path)[0] +"/" +"target_0.model"
        print("self.name_or_path", self.name_or_path)
        with open(self.name_or_path, 'wb') as f:
        # with open(os.path.split(self.name_or_path)[0] +"/" +"target_0.model", 'wb') as f:
            f.write(m.SerializeToString())
                        
        with open(os.path.split(self.name_or_path)[0] +"/"+ os.path.split(self.name_or_path)[-1].split(".")[0] +".vocab", "w") as f:
        # with open(os.path.split(self.name_or_path)[0] +"/"+ "target_0.vocab", "w") as f:
            for item in new_vocab:
                f.write(item[0] + "\t" + str(round(item[1],5)) + "\n")
        
        self.sp = spm.SentencePieceProcessor(model_file=self.name_or_path)
        self.norm_align_pos(according_vocab_length - len(new_pieces))
        
        
        if control_tokens:
            for name, token in control_tokens.items():
                setattr(self, name, token)  
                if self.sp.is_unknown(self.sp.piece_to_id(token)):
                    self.add_user_defined_tokens([token])
        if unknown_token:
            for name, token in unknown_token.items():
                setattr(self, name, token)      
        if os.path.exists(self.special_tokens_map_file):
            special_tokens = json.load(open(self.special_tokens_map_file, "r"))
        else:
            special_tokens = {}
        special_tokens.update(control_tokens)
        special_tokens.update(unknown_token)
        json.dump(special_tokens, open(self.special_tokens_map_file, "w"))
        
        self.additional_special_tokens = special_tokens.get("additional_special_tokens", [])
        
        for name, token in special_tokens.items():
            setattr(self, name, token)
        

        
        
        
        
        
        