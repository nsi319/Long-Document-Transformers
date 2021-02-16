# LongFormer-Models


## Script
To convert bart to long-bart (increased attention window and positional embeddings):

```bash
python bart-longformer/bart_to_longbart.py \
    --base_model facebook/bart-large \
    --tokenizer_name_or_path facebook/bart-large \
    --save_model_path output/long-bart-large-4096 \
    --attention_window 1024 \
    --max_length 4096 
```
To see all the possible command line options, run:

```bash
python bart-longformer/bart_to_longbart.py --help
```
