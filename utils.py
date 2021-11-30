import random
from custom_dataset import load_Dataset
def read_text(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()

        input_seqs, output_seqs = [], []
        for line in lines:
            if line.strip() != '':
                # The file should have tab delimited two columns.
                # First column indicates label field,
                # and second column indicates text field.
                input_seq, output_seq = line.strip().split('\t')
                input_seqs += [input_seq]
                output_seqs += [output_seq]

    return input_seqs, output_seqs

def get_datasets(fn):
     # Get list of labels and list of texts.
    input_seq, output_seq = read_text(fn)

    # Shuffle before split into train and validation set.
    shuffled = list(zip(input_seq, output_seq))
    random.shuffle(shuffled)
    input_seq = [e[0] for e in shuffled]
    output_seq = [e[1] for e in shuffled]

    dataset = load_Dataset(input_seq, output_seq)
    
    return dataset

# def get_datasets(fn, valid_ratio=.2):
#      # Get list of labels and list of texts.
#     input_seq, output_seq = read_text(fn)

#     # Shuffle before split into train and validation set.
#     shuffled = list(zip(input_seq, output_seq))
#     random.shuffle(shuffled)
#     input_seq = [e[0] for e in shuffled]
#     output_seq = [e[1] for e in shuffled]
#     idx = int(len(input_seq) * (1 - valid_ratio))

#     train_dataset = load_Dataset(input_seq[:idx], output_seq[:idx])
#     valid_dataset = load_Dataset(input_seq[idx:], output_seq[idx:])

#     return train_dataset, valid_dataset