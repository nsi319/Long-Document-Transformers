# Long Document Transformers 

Transformers (4.3.0) has  **Longformer Encoder Decoder (LED)** models which is based on BART's architecture and supports **long document generative sequence-to-sequence tasks**.

```LEDForConditionalGeneration``` is an extension of ```BartForConditionalGeneration``` exchanging the traditional self-attention layer with Longformer’s chunked self-attention layer. 

```LEDTokenizer``` is an alias of ```BartTokenizer```. 

* Pre-trained LED models: 
    * ```allenai/led-base-16384```
    * ```allenai/led-large-16384```

Checkout [pre-trained models](https://huggingface.co/models) to see the checkpoints available for each of them.

## Script 

### Finetuning and evaluating LED models on summarization task

We will be using the finetuning script [run.py](https://github.com/nsi319/Finetune-Transformers/blob/main/run.py) which leverages ```Seq2SeqTrainer```. For more information regarding sequence-to-sequence finetuning, visit [Finetune-Transformers](https://github.com/nsi319/Finetune-Transformers). 

For finetuning LED models, set ```--model_name_or_path``` to ```allenai/led-base-16384``` or ```allenai/led-large-16384``` to summarize documents of max length 16,384 tokens.

Here is a sample code to finetune ```allenai/led-large-16384``` for summarization on [`legal dataset`](https://github.com/nsi319/LongFormer-Models/blob/main/dataset/2018_final.csv):

```bash
python run.py  \
    --model_name_or_path allenai/led-large-16384 \
    --train_file dataset/train.csv \
    --validation_file dataset/valid.csv \
    --output_dir legal-led-outputs/new-legal-large-16384-model \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --task summarization \
    --max_source_length=5120 \
    --max_target_length=550 \
    --num_beams=3 \
    --min_summ_length=350 \
    --max_summ_length=500 \
    --length_penalty=2.0 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --predict_with_generate \
    --text_column Text \
    --summary_column Summary 
```

If you are using **Google Colab**, Open [`led-finetune.ipynb`](https://github.com/nsi319/LongFormer-Models/blob/main/led_finetune.ipynb) in Colab, save a copy in Drive and follow the instructions. 

### Converting BART to LONGBART

To convert bart to long-bart (longformer encoder decoder bart) (using LongformerSelfAttention, increased attention window and positional embeddings):

```bash
python convert_bart_to_longbart.py \
    --base_model facebook/bart-large \
    --tokenizer_name_or_path facebook/bart-large \
    --save_model_path output/long-bart-large-4096 \
    --attention_window 1024 \
    --max_length 4096 
```
To see all the possible command line options, run:

```bash
python convert_bart_to_longbart.py --help
```
If you are using **Google Colab**, Open [`run_longbart.ipynb`](https://github.com/nsi319/LongFormer-Models/blob/main/run_longbart.ipynb) in Colab, save a copy in Drive and follow the instructions.
