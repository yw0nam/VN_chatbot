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