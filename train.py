import argparse
import random
import torchtext
from torchtext.legacy import data
from GPT_dataset import *
from utils import read_text
from transformers import T5Tokenizer, AutoModelForCausalLM
from transformers import Trainer
from transformers import TrainingArguments
import torch
import json

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    p.add_argument('--pretrained_model_name', type=str, default='rinna/japanese-gpt2-medium')
    p.add_argument('--gradient_accumulation_steps', type=int, default=4)
    p.add_argument('--valid_ratio', type=float, default=.2)
    p.add_argument('--batch_size_per_device', type=int, default=48)
    p.add_argument('--n_epochs', type=int, default=5)

    p.add_argument('--warmup_ratio', type=float, default=.2)

    p.add_argument('--max_length', type=int, default=512)

    config = p.parse_args()

    return config


def get_datasets(fn, valid_ratio=.2):
     # Get list of labels and list of texts.
    input_seq, output_seq = read_text(fn)

    # Shuffle before split into train and validation set.
    shuffled = list(zip(input_seq, output_seq))
    random.shuffle(shuffled)
    input_seq = [e[0] for e in shuffled]
    output_seq = [e[1] for e in shuffled]
    idx = int(len(input_seq) * (1 - valid_ratio))

    train_dataset = GPTDataset(input_seq[:idx], output_seq[:idx])
    valid_dataset = GPTDataset(input_seq[idx:], output_seq[idx:])

    return train_dataset, valid_dataset


def main(config):
    # Get pretrained tokenizer.
    tokenizer = T5Tokenizer.from_pretrained(config.pretrained_model_name)
    tokenizer.do_lower_case = True
    with open('./data/special_token.json') as f:
        special_tokens = json.load(f)
    # Get datasets and index to label map.
    train_dataset, valid_dataset = get_datasets(
        config.train_fn,
        valid_ratio=config.valid_ratio
    )

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
    model = AutoModelForCausalLM.from_pretrained(
        config.pretrained_model_name
    )
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    training_args = TrainingArguments(
        output_dir='./checkpoints',
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
        warmup_steps=n_warmup_steps,
        weight_decay=0.01,
        fp16=True,
        evaluation_strategy='epoch',
        logging_steps=n_total_iterations // 100,
        save_steps=n_total_iterations // config.n_epochs,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=GPTCollator(tokenizer,
                                  config.max_length,
                                  with_text=False),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    trainer.train()
    
    trainer.model.save_pretrained('./model/yuzubot_context')
    torch.save({
        # 'model': trainer.model.state_dict(),
        'config': config,
        # 'vocab': None,
        # 'classes': None,
        # 'tokenizer': tokenizer,
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
