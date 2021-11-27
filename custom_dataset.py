import torch
from torch.utils.data import Dataset

class dataCollator():

    def __init__(self, tokenizer, max_length, with_text=True, model_type='GPT2'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text
        self.model_type = model_type
        
    def __call__(self, samples):
        input_seq = [s['input_seq'] for s in samples]
        output_seq = [s['output_seq'] for s in samples]
        
        if self.model_type == 'BART':
            input_encoding = self.tokenizer(
                input_seq,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length
            )
            
            output_encoding = self.tokenizer(
                output_seq,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length
            )
            # labels = output_encoding['input_ids']
            # labels = [
            #     [(label if label != self.tokenizer.pad_token_id else -100) for label in labels_example] for labels_example in labels
            # ]
            # labels = torch.tensor(labels)
            return_value = {
                'input_ids': input_encoding['input_ids'],
                'attention_mask': input_encoding['attention_mask'],
                'labels': output_encoding['input_ids'],
            }
        elif self.model_type == 'GPT2':
            seq = [input_seq+'</s>'+output_seq for input_seq, output_seq in zip(input_seq,output_seq)]
            encoding = self.tokenizer(
                seq,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length
            )
            return_value = {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': encoding['input_ids']
            }
        if self.with_text:
            return_value['input_seq'] = input_seq
            return_value['output_seq'] = output_seq

        return return_value


class load_Dataset(Dataset):

    def __init__(self, input_seq, output_seq):
        self.input_seq = input_seq
        self.output_seq = output_seq
    
    def __len__(self):
        return len(self.input_seq)
    
    def __getitem__(self, item):
        input_seq = str(self.input_seq[item])
        output_seq = str(self.output_seq[item])

        return {
            'input_seq': input_seq,
            'output_seq': output_seq,
        }