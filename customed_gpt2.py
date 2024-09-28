import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import copy
import json


import torch
import torch.utils.checkpoint
from torch import nn
from transformers.pytorch_utils import Conv1D
from torch.nn.functional import linear
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    HfArgumentParser,
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    Trainer,
    GPT2Model,
    GPT2PreTrainedModel,
    default_data_collator,
    is_torch_tpu_available,
    TrainingArguments,
    GPT2LMHeadModel,
    set_seed,
)
from transformers.utils import logging
from torch.nn.parameter import Parameter, UninitializedParameter
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.generation import GenerationMixin

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)


class Plug_LM_Header(torch.nn.Module):
    r"""
    This module implements lm_header that enables hidden_states transformed to incremental vocabulary in a plug-and-play way.
    """
    def __init__(self, config, num_incremental_embeddings: Optional[int] = 0, _weight: Optional[torch.Tensor] = None,
                 device=None, dtype=None, bias=None) -> None:
        '''
        num_embeddings: output size of lm head, which should be the size of target vocab
        
        num_incremental_embeddings: if output size is larger than input size of embedding layer, that incremetnal size should 
        be the reduction of target vocab size and source vocab size. It can be 0 if there is no index overlap between source and target vocabulary.
        
        mapping_index_file: if not none, a cross-attention mechanism of the incremental weight over source weight will be used. 
        
        
        '''
        factory_kwargs = {'device': device, 'dtype': dtype, 'bias': bias}
        self.config = config
        self.out_features = config.vocab_size
        self.in_features = config.n_embd if hasattr(config,  "n_embd") else config.d_model
        super(Plug_LM_Header, self).__init__()
        self.num_incremental_embeddings = num_incremental_embeddings
        # self.lazy_mapping_weight = config.lazy_mapping_weight
        if num_incremental_embeddings > 0:
            self.c_attn = Conv1D(2 * self.in_features, self.in_features)
            self.q_attn = Conv1D(self.in_features, self.in_features)
            self.c_proj = Conv1D(self.in_features, self.in_features)
            self.resid_dropout = nn.Dropout(config.resid_pdrop if  hasattr(config, "resid_pdrop") else config.dropout)
            self.attn_dropout = nn.Dropout(config.attn_pdrop if hasattr(config,  "attn_pdrop") else config.attention_dropout)
        self.device = device
        self.dtype = dtype
        self.mapping_index = None
        if _weight is None:
            self.lm_head = torch.nn.Linear(self.in_features, self.out_features, **factory_kwargs)
            print("self.lm_head size:", self.lm_head.weight.size(), "device:", self.lm_head.weight.device)
            # self.weight = Parameter(torch.empty((self.out_features - num_incremental_embeddings, self.in_features), **factory_kwargs))
            # self.lm_head = torch.nn.Linear(self.in_features, self.out_features - self.num_incremental_embeddings, **factory_kwargs)
            # if self.num_incremental_embeddings > 0:
            #     # self.weight_incremental = Parameter(torch.empty((num_incremental_embeddings, self.in_features), **factory_kwargs))
            #     self.incremental_lm_head = torch.nn.Linear(self.in_features, self.num_incremental_embeddings, **factory_kwargs)
            #     print("self.incremental_lm_head size:", self.incremental_lm_head.weight.size())
        else:
            assert list(_weight.shape) == [self.out_features, self.in_features], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.lm_head = torch.nn.Linear(self.in_features, self.out_features, **factory_kwargs)
            # self.lm_head = torch.nn.Linear(self.in_features, self.out_features - self.num_incremental_embeddings, **factory_kwargs)
            self.lm_head.weight = Parameter(_weight)
           
        if config.mapping_index_file:
            mapping_index_matrix = json.load(open(config.mapping_index_file, "r"))
            self.mapping_index = torch.Tensor(mapping_index_matrix).to(self.device).to(torch.long)
            self.mapping_lm_head = torch.nn.Linear(self.in_features, self.out_features, **factory_kwargs) 
            # self.tie_mapping_lm_head(self.dtype, self.device)              

            
                
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )
               

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights
    
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)
    
    def cross_attention(self, query_embeds, kv_embeds, num_heads=4, attention_mask=None, output_attentions=False):
        self.num_heads = num_heads
        self.head_dim = self.in_features // self.num_heads
        query = self.q_attn(query_embeds)
        key, value = self.c_attn(kv_embeds).split(self.in_features, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask=attention_mask)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs
    
    def get_mapping_weight(self, dtype, device):
        if not hasattr(self, "mapping_index") or self.mapping_index is None:
            return
            
        device = self.lm_head.weight.device
        q = self.lm_head.weight[-self.num_incremental_embeddings:, None, :]
        kv_table = self.lm_head.weight
        kv = torch.clone(kv_table[self.mapping_index].detach())
        kv.requires_grad = False
        
        attention_mask = torch.not_equal(self.mapping_index, self.config.pad_token_id).to(device).to(torch.long)
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(device)
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(dtype if dtype is not None else torch.float).min
        outputs = self.cross_attention(q, kv, attention_mask=attention_mask)
        
        weight_incremental_after_attention = outputs.view(-1, self.in_features)
        # weight = torch.concat([self.lm_head.weight,weight_incremental_after_attention], dim = 0)
        weight = torch.concat([self.lm_head.weight[:-self.num_incremental_embeddings, :], weight_incremental_after_attention], dim = 0)
        
        # self.mapping_lm_head.weight is used for parameter storing. However parameter is not in computation graph, for gradient prop, we use variable weight to calculate loss.
        return weight
    
    def tie_mapping_lm_head(self, dtype, device):
        if not hasattr(self, "mapping_index") or self.mapping_index is None:
            return
        
        weight = self.get_mapping_weight(dtype, device)
        
        self.mapping_lm_head.weight = nn.Parameter(weight)
        
        
        
    def init_lm_head_by_mapping(self, dtype, device):
        if not hasattr(self, "mapping_index") or self.mapping_index is None:
            return
        # logger.info("init_lm_head_by_mapping")
        kv_table = self.lm_head.state_dict()
        # logger.info(f"{(kv_table['weight'])[-1:,:]}")
        kv = kv_table["weight"][self.mapping_index]
        # logger.info(f"{kv[-1:, :, :].size()}")
        # logger.info(self.mapping_index.size())
        # logger.info(f"{self.mapping_index[-1:, :]}")
        # logger.info(self.config.pad_token_id)
        attention_mask = torch.concat([torch.ones_like(self.mapping_index[:, :1], dtype=dtype), torch.not_equal(self.mapping_index[:, 1:], self.config.pad_token_id).to(dtype=dtype)], dim=1)
        # logger.info(f"{attention_mask[-1:,:]}")
        # logger.info(f"{torch.mm(attention_mask[-1:,:],kv[-1,:,:])[0,:10] }")
        # logger.info(kv.permute(0, 2, 1).size())
        # logger.info(attention_mask.size())
        # logger.info(f"{torch.bmm(kv.permute(0, 2, 1), attention_mask[:,:,None]).size()}") 
        # logger.info(f"{(torch.matmul(attention_mask, kv))[:1,:]}")
        # logger.info(f"{torch.sum(attention_mask, 1).size()}")
        
        # logger.info(f"{(torch.bmm(kv.permute(0, 2, 1), attention_mask[:,:,None]))[-1, :10, 0]}")
     
        kv_table["weight"][-self.num_incremental_embeddings:, :] = torch.div(torch.bmm(kv.permute(0, 2, 1), attention_mask[:,:,None]), torch.sum(attention_mask.long(), 1)[:, None, None]).squeeze(2)    
        # logger.info(f"{kv_table['weight'][-1:, :]}")
        self.lm_head.load_state_dict(kv_table)
        
        
        
    def forward(self, hidden_states: torch.Tensor, mapping_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Apply a cross-attention between the weights of incremental vocabulary and the weights of their corresponding subwords (containing themselves).
        # mapping_index stores the indices of the corresponding subwords of each word in the incremental vocabulary. 
        # mapping_index size: (num_incremental_embeddings, num_subwords)
        if mapping_index is not None:
            self.mapping_index = mapping_index
            
        logits = self.lm_head(hidden_states)
            
        # if self.mapping_index is not None:
        #     if not self.lazy_mapping_weight:
        #         weight = self.get_mapping_weight(hidden_states.dtype, hidden_states.device)
        #         logits = torch.nn.functional.linear(hidden_states, weight)               
        #     else:
        #         # if use lazy, parameter of lm_head is staticly stored in self.mapping_lm_head
        #         logits = self.mapping_lm_head(hidden_states)
        # else:  
        #     logits = self.lm_head(hidden_states)
        #     # logger.info("Out mapping index")                    
        
        return logits
    


class CusVocab_GPT2LMHeadModel(GPT2PreTrainedModel, GenerationMixin):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        # self.lm_head = nn.Linear(config.n_embd, config.output_vocab_size, bias=False)
        incremental_vocab_size = config.vocab_size - config.base_vocab_size
        print("incremental_vocab_size:", incremental_vocab_size)
        '''
        Plu_LM_Header supports three types of lm_header_weight calculation:
        if incremental vocabulary size > 0 and mapping_index is given, get lm head weight by applying cross-attention between incremental vocab and their subwords.
        if incremental vocabulary size > 0 and mapping_index is None, learn incremental part of lm head weight from scratch, without utilizing information of subwords.
        if incremental vocabulary size = 0, do not allow a customed vocabulary, which is the same as the primary lm head layer.
        '''
        self.customed_lm_head = Plug_LM_Header(config, incremental_vocab_size)
    

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()
     
        

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.customed_lm_head = self.customed_lm_head.to(self.transformer.first_device)
        self.model_parallel = True
        
        
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
            

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
           
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }


    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.customed_lm_head = self.customed_lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.customed_lm_head.lm_head

    def set_output_embeddings(self, embeddings):
        self.customed_lm_head.lm_head.weight = embeddings

        
    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""           
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        mapping_matrix: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict   
        # logger.level = logging.get_verbosity()
        # logger.setLevel(logging.get_verbosity())
        # logger.info("level: %s%s"%(logger.level,logging.get_verbosity()) )
        # logger.info("customed_gpt2 input_ids:%s"%input_ids[0])
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = transformer_outputs[0]
        # logger.info("customed_gpt2 wte:%s"%self.transformer.wte.weight[0, :10])
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
            
        # Modified:
        lm_logits = self.customed_lm_head(hidden_states)
        # logger.info("customed_gpt2 lm_logits:%s"%lm_logits[0, 0, :10])
        loss = None  
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # logger.info(f"get loss, device{loss.device}")
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        # logger.info("return model")
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )