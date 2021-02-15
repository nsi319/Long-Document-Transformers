import argparse
import logging
import os

from transformers import BartTokenizer

from transformers import BartForConditionalGeneration
from transformers.modeling_bart import shift_tokens_right
from longformer.longformer_encoder_decoder import LongformerSelfAttentionForBart, LongformerEncoderDecoderConfig
from longformer.longformer_encoder_decoder import LongformerEncoderDecoderForConditionalGeneration

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def convert_to_long_model(
    save_model_path, 
    base_model='facebook/bart-large',
    tokenizer_name_or_path='facebook/bart-large',
    max_length=4096,
    attention_window=1024,
):
    model = BartForConditionalGeneration.from_pretrained(base_model)
    tokenizer = BartTokenizer.from_pretrained(tokenizer_name_or_path, model_max_length=max_length)
    config = LongformerEncoderDecoderConfig.from_pretrained(base_model)
    model.config = config

    # in BART attention_probs_dropout_prob is attention_dropout, but LongformerSelfAttention
    # expects attention_probs_dropout_prob, so set it here
    config.attention_probs_dropout_prob = config.attention_dropout
    config.architectures = ['LongformerEncoderDecoderForConditionalGeneration', ]

    # extend position embeddings
    tokenizer.model_max_length = max_length
    tokenizer.init_kwargs['model_max_length'] = max_length
    current_max_length, embed_size = model.model.encoder.embed_positions.weight.shape
    assert current_max_length == config.max_position_embeddings + 2

    config.max_encoder_position_embeddings = max_length
    config.max_decoder_position_embeddings = config.max_position_embeddings
    del config.max_position_embeddings
    max_length += 2  # NOTE: BART has positions 0,1 reserved, so embedding size is max position + 2
    assert max_length >= current_max_length

    # allocate a larger position embedding matrix for the encoder
    new_encoder_pos_embed = model.model.encoder.embed_positions.weight.new_empty(max_length, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_length - 2
    while k < max_length - 1:
        new_encoder_pos_embed[k:(k + step)] = model.model.encoder.embed_positions.weight[2:]
        k += step
    model.model.encoder.embed_positions.weight.data = new_encoder_pos_embed

    # allocate a larger position embedding matrix for the decoder
    # new_decoder_pos_embed = model.model.decoder.embed_positions.weight.new_empty(max_length, embed_size)
    # # copy position embeddings over and over to initialize the new position embeddings
    # k = 2
    # step = current_max_length - 2
    # while k < max_length - 1:
    #     new_decoder_pos_embed[k:(k + step)] = model.model.decoder.embed_positions.weight[2:]
    #     k += step
    # model.model.decoder.embed_positions.weight.data = new_decoder_pos_embed

    # replace the `modeling_bart.SelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    config.attention_dilation = [1] * config.num_hidden_layers

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


def main():
    parser = argparse.ArgumentParser(description="Convert BART to LongBART. Replaces BART encoder's SelfAttnetion with LongformerSelfAttention")
    parser.add_argument(
        '--base_model',
        type=str,
        default='facebook/bart-large',
        help='The name or path of the base model you want to convert'
    )
    parser.add_argument(
        '--tokenizer_name_or_path',
        type=str,
        default='facebook/bart-large',
        help='The name or path of the tokenizer'
    )
    parser.add_argument(
        '--save_model_path',
        type=str,
        required=True,
        help='The path to save the converted model'
    )
    parser.add_argument(
        '--attention_window',
        type=int,
        default=512,
        help='attention window size for longformer self attention (one sided)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=4096 * 4,
        help='maximum encoder positions'
    )

    args = parser.parse_args()

    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)

    convert_to_long_model(
        save_model_path=args.save_model_path,
        base_model=args.base_model,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        attention_window=args.attention_window,
        max_length=args.max_length
    )

if __name__ == "__main__":
    main()