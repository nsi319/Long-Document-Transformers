### Transformer version: 2.10.0

import argparse
import logging
import os

from transformers import BartTokenizer,HfArgumentParser

from modeling_bart import BartForConditionalGeneration
from modeling_longbart import LongformerSelfAttentionForBart 

from dataclasses import dataclass, field
from typing import Optional


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def create_long_model(
    save_model_path, 
    base_model='facebook/bart-large',
    tokenizer_name_or_path='facebook/bart-large',
    max_length=4096,
    attention_window=1024,
):
    model = BartForConditionalGeneration.from_pretrained(base_model)
    tokenizer = BartTokenizer.from_pretrained(tokenizer_name_or_path, model_max_length=max_length)
    config = model.config

    # in BART attention_probs_dropout_prob is attention_dropout, but LongformerSelfAttention
    # expects attention_probs_dropout_prob, so set it here  
    config.attention_probs_dropout_prob = config.attention_dropout

    # extend position embeddings
    tokenizer.model_max_length = max_length
    tokenizer.init_kwargs['model_max_length'] = max_length
    current_max_length, embed_size = model.model.encoder.embed_positions.weight.shape
    # config.max_position_embeddings = max_pos
    config.encoder_max_position_embeddings = max_length
    max_length += 2  # NOTE: BART has positions 0,1 reserved, so embedding size is max position + 2
    assert max_length > current_max_length
    # allocate a larger position embedding matrix
    new_pos_embed = model.model.encoder.embed_positions.weight.new_empty(max_length, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_length - 2
    while k < max_length - 1:
        new_pos_embed[k:(k + step)] = model.model.encoder.embed_positions.weight[2:]
        k += step
    model.model.encoder.embed_positions.weight.data = new_pos_embed

    # replace the `modeling_bart.SelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.model.encoder.layers):
        longformer_self_attn_for_bart = LongformerSelfAttentionForBart(config, layer_id=i)
        
        longformer_self_attn_for_bart.longformer_self_attn.query = layer.self_attn.q_proj
        longformer_self_attn_for_bart.longformer_self_attn.key = layer.self_attn.k_proj
        longformer_self_attn_for_bart.longformer_self_attn.value = layer.self_attn.v_proj

        longformer_self_attn_for_bart.longformer_self_attn.query_global = layer.self_attn.q_proj
        longformer_self_attn_for_bart.longformer_self_attn.key_global = layer.self_attn.k_proj
        longformer_self_attn_for_bart.longformer_self_attn.value_global = layer.self_attn.v_proj

        longformer_self_attn_for_bart.output = layer.self_attn.out_proj

        layer.self_attn = longformer_self_attn_for_bart

    logger.info(f'saving model to {save_model_path}')
    model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)
    return model, tokenizer



@dataclass
class ModelArguments:
    """
    Arguments pertaining to model parameters.
    """
    save_model_path: str = field(
        metadata={
        "help":'The path to save the converted model'
        },
    )
    base_model: Optional[str] = field(
        default="facebook/bart-large",
        metadata={
        "help":'The name or path of the base model you want to convert'
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default="facebook/bart-large",
        metadata={
        "help":'The name or path of the tokenizer'
        },
    )
    max_length: Optional[int] = field(
        default=4096,
        metadata={
        "help":'Maximum encoder positions'
        },
    )
    attention_window: Optional[int] = field(
        default=1024,
        metadata={
        "help":'Attention window size for longformer self attention'
        },
    )

def main():
    parser = HfArgumentParser((ModelArguments))
    args = parser.parse_args_into_dataclasses()[0]

    print(args)
    print(type(args))
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)
    
    create_long_model(
        base_model=args.base_model,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        save_model_path=args.save_model_path,
        max_length=args.max_length,
        attention_window=args.attention_window,
    )


if __name__ == "__main__":
    main()