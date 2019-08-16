import string
import random

char_to_ix = string.printable


def get_index(c):
    if c in string.digits:
        return char_to_ix.index('0')

    return char_to_ix.index(c)


def prepare_sequence(str, chunck_len):
    seq = [get_index(c) for c in str.lower() if c in char_to_ix and c and c not in string.punctuation]
    l = len(seq)
    if l < chunck_len:
        seq = seq + [0 for x in range(0, chunck_len - l)]
    assert len(seq) == chunck_len
    return seq


def batch(x, y, n=1):
    l = len(x)
    batches = []
    for ndx in range(0, l, n):
        batches.append((x[ndx:min(ndx + n, l)], y[ndx:min(ndx + n, l)]))
    return batches


def get_random_samples(data_set, sample_len, n):
    D = []
    for (str, label) in data_set:
        for i in range(0, n):
            start_index = random.randint(0, len(str) - sample_len)
            end_index = start_index + sample_len
            D.append((prepare_sequence(str[start_index:end_index], sample_len), label))

    random.shuffle(D)
    X = [x[0] for x in D]
    Y = [x[1] for x in D]
    return X, Y


def read_dir(dir):
    train_set = []
    for file in dir.iterdir():
        l = 1 if file.name.endswith("Y.txt") else 0
        with file.open() as f:
            train_set.append((f.read(), l))

    return train_set
