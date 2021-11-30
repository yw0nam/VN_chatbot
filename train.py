import argparse
import random
import torchtext
from torchtext.legacy import data
from custom_dataset import *
from utils import *
from transformers import Trainer
from transformers import TrainingArguments
import torch
import json
from transformers import AutoModelForCausalLM
def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    p.add_argument('--valid_fn', required=True)
    p.add_argument('--gradient_accumulation_steps', type=int, default=2)
    p.add_argument('--valid_ratio', type=float, default=.2)
    p.add_argument('--batch_size_per_device', type=int, default=48)
    p.add_argument('--n_epochs', type=int, default=5)
    p.add_argument('--model', type=str, default='gpt2')
    p.add_argument('--model_type', type=str, default='causal_lm')
    p.add_argument('--warmup_ratio', type=float, default=.2)
    p.add_argument('--max_length', type=int, default=512)
    p.add_argument('--load_weight', default=None)

    config = p.parse_args()

    return config

def main(config):
    # Get pretrained tokenizer.
    with open('./data/special_token.json') as f:
        special_tokens = json.load(f)
        
    if config.model == 'gpt2':
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained('rinna/japanese-gpt2-medium')
        model = AutoModelForCausalLM.from_pretrained('rinna/japanese-gpt2-medium')
    elif config.model == 'mbart':
        from transformers import MBartTokenizer
        tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25",
                                                 src_lang="ja_XX", tgt_lang="ja_XX")
        if config.load_weight:
            model = AutoModelForCausalLM.from_pretrained(config.load_weight)
        else:
            model = AutoModelForCausalLM.from_pretrained("facebook/mbart-large-cc25")
        special_tokens['additional_special_tokens'].append('ja_XX')
    
    tokenizer.do_lower_case = True
    # Get datasets and index to label map.
#     train_dataset, valid_dataset = get_datasets(
#         config.train_fn,
#         valid_ratio=config.valid_ratio
#     )
    train_dataset = get_datasets(config.train_fn)
    valid_dataset = get_datasets(config.valid_fn)
    print(
        '|train| =', len(train_dataset),
        '|valid| =', len(valid_dataset),
    )

    total_batch_size = config.batch_size_per_device * torch.cuda.device_count()
    n_total_iterations = int(len(train_dataset) / total_batch_size * config.n_epochs)
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )

    # Get pretrained model with specified softmax layer.
    tokenizer.add_special_tokens(special_tokens)
    if not config.load_weight:
        model.resize_token_embeddings(len(tokenizer))
    
    training_args = TrainingArguments(
        output_dir='./model/checkpoints',
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
        warmup_steps=n_warmup_steps,
#         weight_decay=0.01,
        fp16=True,
        evaluation_strategy='epoch',
        logging_steps=n_total_iterations // 100,
        save_strategy ='epoch',
#         save_steps=n_total_iterations // config.n_epochs,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        load_best_model_at_end=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=dataCollator(tokenizer,
                                  config.max_length,
                                  with_text=False,
                                  model_type=config.model_type),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    trainer.train()
    
    trainer.model.save_pretrained(config.model_fn)
    torch.save({
        # 'model': trainer.model.state_dict(),
        'config': config,
        # 'vocab': None,
        # 'classes': None,
        # 'tokenizer': tokenizer,
    }, config.model_fn + 'train_config.json')


if __name__ == '__main__':
    config = define_argparser()
    main(config)
