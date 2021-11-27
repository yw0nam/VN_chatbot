# %%
import random
import torchtext
from torchtext.legacy import data
from custom_dataset import *
from utils import *
from transformers import T5Tokenizer, AutoModelForCausalLM
from transformers import Trainer
from transformers import TrainingArguments
import torch
from torch.utils.data import DataLoader
import re
import pandas as pd
import json
# %%
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, MBartTokenizer, MBartForCausalLM

# %%
# %%
tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')
# tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-cc25",
                                                 src_lang="ja_XX", tgt_lang="ja_XX")
tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
# %%
with open('./data/special_token.json') as f:
    special_tokens = json.load(f)
    special_tokens['additional_special_tokens'].append('ja_XX')
tokenizer.add_special_tokens(special_tokens)
# %%
# model = AutoModelForCausalLM.from_pretrained('./model//yuzubot_context/').cuda()
# %%
train_dataset, valid_dataset = get_datasets(
    fn='./data/QA.tsv',
    valid_ratio=0.1
)

# %%
temp = []
idx = 0
for i in valid_dataset:
    temp.append(i)
    idx += 1
    if idx == 5:
        break
# %%
collator = dataCollator(tokenizer, 512, model_type='BART')
data = collator(temp)
# %%
# This code from https://huggingface.co/ushikado/yuyuyui-chatbot.
class Interlocutor():
    def __init__(self, tokenizer, model, character_token, max_context_length=512, max_response_length=128):
        self.tokenizer = tokenizer
        self.model = model
        self.character_token = character_token
        self.max_context_length = max_context_length
        self.max_response_length = max_response_length
        self.context = ""
        return
    def generate_with_character(self, query, chara_idx):
        nanigashi = self.tokenizer.additional_special_tokens[chara_idx]
        nanigashi_id = self.tokenizer.additional_special_tokens_ids[chara_idx]
        self.context += nanigashi + query + self.tokenizer.eos_token + self.character_token
        context_tensor = self.tokenizer.encode(self.context, add_special_tokens=False, return_tensors="pt").cuda()
        context_length = context_tensor.size()[-1]
        if self.max_context_length < context_length:
            context_tensor = context_tensor.narrow(1, context_length - self.max_context_length, self.max_context_length)
            context_length = context_tensor.size()[-1]
        max_length = context_length + self.max_response_length
        context_tensor = self.model.generate(context_tensor, do_sample=True, max_length=max_length,
                                             pad_token_id=self.tokenizer.eos_token_id)
        self.context = re.sub(self.tokenizer.eos_token, "", self.tokenizer.decode(context_tensor[0]))
        response = self.context[self.context.rindex(self.character_token) + len(self.character_token) : ].strip()
        print(response)
        
    def generate_nanigashi(self, query):
        nanigashi = self.tokenizer.additional_special_tokens[-1]
        nanigashi_id = self.tokenizer.additional_special_tokens_ids[-1]
        self.context += nanigashi + query + self.tokenizer.eos_token + self.character_token
        context_tensor = self.tokenizer.encode(self.context, add_special_tokens=False, return_tensors="pt").cuda()
        context_length = context_tensor.size()[-1]
        if self.max_context_length < context_length:
            context_tensor = context_tensor.narrow(1, context_length - self.max_context_length, self.max_context_length)
            context_length = context_tensor.size()[-1]
        max_length = context_length + self.max_response_length
        context_tensor = self.model.generate(context_tensor, do_sample=True, max_length=max_length,
                                             pad_token_id=self.tokenizer.eos_token_id)
        self.context = re.sub(self.tokenizer.eos_token, "", self.tokenizer.decode(context_tensor[0]))
        response = self.context[self.context.rindex(self.character_token) + len(self.character_token) : ].strip()
        print(response)
# %%
interlocutor = Interlocutor(tokenizer, model, "<ナツメ>")
interlocutor.generate_with_character("今時間大丈夫？", -1)
# %%
