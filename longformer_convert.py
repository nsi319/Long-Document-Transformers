import argparse
import logging
import os
import copy
import torch

from dataclasses import dataclass, field
from typing import Optional

from transformers import BartTokenizer,HfArgumentParser
from transformers import BartForConditionalGeneration
from transformers.modeling_bart import shift_tokens_right
from longformer.longformer_encoder_decoder import LongformerSelfAttentionForBart, LongformerEncoderDecoderConfig
from longformer.longformer_encoder_decoder import LongformerEncoderDecoderForConditionalGeneration

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
    config.attention_dilation = [1] * config.num_hidden_layers

    for i, layer in enumerate(model.model.encoder.layers):
        longformer_self_attn_for_bart = LongformerSelfAttentionForBart(config, layer_id=i)

        longformer_self_attn_for_bart.longformer_self_attn.query = layer.self_attn.q_proj
        longformer_self_attn_for_bart.longformer_self_attn.key = layer.self_attn.k_proj
        longformer_self_attn_for_bart.longformer_self_attn.value = layer.self_attn.v_proj

        longformer_self_attn_for_bart.longformer_self_attn.query_global = copy.deepcopy(layer.self_attn.q_proj)
        longformer_self_attn_for_bart.longformer_self_attn.key_global = copy.deepcopy(layer.self_attn.k_proj)
        longformer_self_attn_for_bart.longformer_self_attn.value_global = copy.deepcopy(layer.self_attn.v_proj)

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

    tokenizer = BartTokenizer.from_pretrained(args.save_model_path)
    model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(args.save_model_path)
    model.model.encoder.config.gradient_checkpointing = True
    model.model.decoder.config.gradient_checkpointing = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    padding = "max_length" 

    text="""The Securities and Exchange Commission charged William "Bill" Bowser, Christopher Ashby, Scott Beynon, and Jordan Nelson with securities fraud for misappropriating investor funds meant for the development and construction of new event centers. The SEC's complaint alleges that from approximately January 2017 to February 2019, Ashby, Beynon, and Nelson, through entities they controlled, sold investors interests in for-profit event centers purportedly being developed by Noah Corporation, an entity Bowser controlled. As alleged, Bowser diverted investor funds earmarked for specific properties and instead used them for Noah Corporation's and Bowser's operational and other expenses and to pay prior investors, rather than for construction of event centers as represented to investors. The complaint further alleges that contrary to their representations to investors, Ashby, Beynon, and Nelson failed to escrow investor funds and disbursed them to an entity Bowser controlled without having any controls in place to ensure that the disbursements were for legitimate expenses. The SEC's complaint, filed in federal district court in Utah, charges Bowser with violating the antifraud provisions of Sections 17(a)(1) and 17(a)(3) of the Securities Act of 1933 and Section 10(b) of the Securities Exchange Act of 1934 and Rules 10b-5(a) and (c) thereunder. The complaint charges Ashby, Beynon, and Nelson with violating the antifraud provisions of Sections 17(a)(2) and 17(a)(3) of the Securities Act and the registration provisions of Section 15(a) of the Exchange Act. Without admitting or denying the allegations of the complaint, the defendants have consented to judgments enjoining them from violating the charged provisions, ordering Bowser to pay disgorgement of $47,796 with prejudgment interest of $6,402 and a $192,768 penalty, ordering Ashby to pay disgorgement of $551,161 with prejudgment interest of $43,994 and a $96,384 penalty, ordering Beynon to pay disgorgement of $585,426 with prejudgment interest of $46,729 and a $96,384 penalty, and ordering Nelson to pay disgorgement of $281,273 with prejudgment interest of $22,451 and a $96,384 penalty. The SEC's investigation was conducted by Cheryl Mori and was supervised by Daniel Wadley and Amy Oliver of the Salt Lake Regional Office. The litigation will be led by Casey Fronk. The Securities and Exchange Commission today announced charges against DeAndre P. Sears and MASears LLC d/b/a Picasso Group, an entity he controlled and operated, with registration violations for unlawfully selling securities of Florida-based real estate firm EquiAlt LLC to retail investors. The SEC previously filed an enforcement action against EquiAlt LLC, its CEO Brian Davison, and its Managing Director Barry Rybicki on February 11, 2020, in connection with the alleged scheme. According to the SEC's complaint, between 2014 and 2020, Sears directly and indirectly, through the use of third-party agents, sold at least $25 million of EquiAlt's securities to more than 145 largely unaccredited, unsophisticated, and elderly retail investors located in 25 states. During that period, Sears was identified in EquiAlt private placement memoranda as Managing Director of Investments, President of Business Development and Marketing, or Vice President of Investor Relations. Sears, through Picasso Group, received approximately $3.5 million in transaction-based sales commissions from EquiAlt, despite neither being registered as broker dealers. The complaint alleges that beginning in approximately 2016, EquiAlt was actually operating a Ponzi scheme during which it raised more than $170 million from approximately 1,100 investors in 35 states. The SEC's complaint charges Sears and Picasso Group with violating the securities registration provisions of Sections 5(a) and 5(c) of the Securities Act of 1933, and the broker-dealer registration provisions of Section 15(a)(1) of the Securities Exchange Act of 1934. Without admitting or denying the allegations in the complaint, Sears and Picasso Group have agreed to the entry of a judgment providing injunctive relief with disgorgement and civil penalties to be determined by a court at a later date. Sears also agreed to associational and penny stock bars as part of a settled follow-on administrative proceeding. The SEC's continuing investigation is being conducted by Chanel T. Rowe and Andre Zamorano, with assistance from Mark Dee, and supervised by Thierry Olivier Desmet and Glenn S. Gordon in the Miami Regional Office. The SEC's litigation is being led by Alise Johnson and supervised by Andrew O. Schiff. The SEC encourages investors to check the backgrounds of people selling investments by using the SEC's Investor.gov to identify quickly whether they are registered professionals and confirm their identity. On December 30, 2020, the U. S. District Court for the Southern District of New York entered final judgments against two penny stock promoters whom the Commission charged in connection with several alleged pump-and-dump schemes involving stocks they were touting in their supposedly independent newsletters. The SEC's complaint in this action, filed in November 2014, alleged that Anthony Thompson, Jr., Jay Fung, and a third defendant, Eric Van Nguyen, worked in concert to gain control of a large portion of shares in the stock of microcap companies, then hyped those stocks in newsletters they distributed to prospective investors. According to the complaint, the newsletters published by Thompson, Fung, and Van Nguyen misleadingly stated that they "may" or "might" sell shares they owned when in reality they always intended to sell -- and in some instances already were selling - the stocks they were promoting. As alleged, they also failed to fully disclose in their newsletters the amounts of compensation they were receiving for promoting the stocks. Thompson and Fung, who were both previously convicted in a parallel state criminal case for conduct that was the subject of the SEC's action, each consented to the entry of a final judgment enjoining them from future violations of the anti-touting provisions of Section 17(b) of the Securities Act of 1933. Thompson further consented to pay disgorgement of $624,882 plus prejudgment interest of $137,381, and Fung consented to pay disgorgement of $1,766,083 plus prejudgment interest of $244,308. Fung's obligation to pay disgorgement and prejudgment interest will be deemed satisfied by the $2,800,000 he was ordered to pay pursuant to an order of restitution entered in a related criminal action filed by the Manhattan District Attorney's Office. The SEC's litigation continues with respect to Van Nguyen. The SEC's litigation has been handled by Peter Pizzani, Mark R. Sylvester, and Thomas P. Smith, Jr., and supervised by Sanjay Wadhwa, all of the New York Regional Office."""
    input_tokenized = tokenizer.encode(text, return_tensors='pt',padding=padding,pad_to_max_length=True, padding_side='right', max_length=4096,truncation=True).to(device)
    
    dim = list(input_tokenized.size())
    print("Input Shape: {} (padded to max_length)".format(dim))

    summary_ids = model.generate(input_tokenized,
                                      num_beams=4,
                                      no_repeat_ngram_size=3,
                                      length_penalty=1.0,
                                      min_length=512,
                                      max_length=2016)
    
    dim = list(summary_ids.size())
    print("Summary Shape: ", dim)
      
    summ = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
    
    print("Text: ")
    print(text)
    print("Summary: ")
    print(summ)


if __name__ == "__main__":
    main()