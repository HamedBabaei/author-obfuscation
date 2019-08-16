import argparse
from pathlib import Path
import random
import shutil

argparser = argparse.ArgumentParser()
argparser.add_argument('corpus', type=str)
argparser.add_argument('--output', type=str, default="./data_obf/")
argparser.add_argument('--obf', type=str, default=True)
args = argparser.parse_args()

corpus_dir = Path(args.corpus)
output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

if not corpus_dir.exists():
    raise Exception("cannot find corpus directory at: " + corpus_dir)

truth_file = corpus_dir / "truth.txt"

if not truth_file.exists() and not args.obf:
    raise Exception("cannot find truth file: " + truth_file)
if args.obf:
    truth = [[x.name, 'Y'] for x in corpus_dir.iterdir()]
else:
    with truth_file.open() as f:
        lines = f.readlines()
        truth = [l.split() for l in lines]


def get_files(name):
    folder = corpus_dir / name
    if args.obf:
        train_files = [file for file in folder.glob("same-author*.txt")]
        test_files = [file for file in folder.glob("original.txt")]
    else:
        train_files = [file for file in folder.glob("known*.txt")]
        test_files = [file for file in folder.glob("unknown*.txt")]

    return train_files, test_files


all = {i for i in range(0, len(truth))}
for i in range(len(truth)):
    (name, label) = truth[i]
    folder = corpus_dir / name
    known, unknown = get_files(name)

    temp = list(all.difference({i}))
    random.shuffle(temp)
    other_known = []
    c = 0
    while len(other_known) < len(known):
        other = temp[c]
        k, _ = get_files(truth[other][0])
        other_known = other_known + k[0: len(known) - len(other_known)]
        c += 1
    assert len(known) == len(other_known)

    train = output_dir / name / "train"
    train.mkdir(parents=True, exist_ok=True)

    for n, file in enumerate(known):
        file_name = "%d-Y.txt" % n
        shutil.copyfile(file, train / file_name)

    for n, file in enumerate(other_known):
        file_name = "%d-N.txt" % (n + len(known))
        shutil.copyfile(file, train / file_name)

    test = output_dir / name / "test"
    test.mkdir(parents=True, exist_ok=True)
    for n, file in enumerate(unknown):
        file_name = "%d-%s.txt" % (n, label)
        shutil.copyfile(file, test / file_name)
