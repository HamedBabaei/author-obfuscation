import torch
import torch.autograd as autograd
import argparse
from pathlib import Path
from utils import *


def test(model, data_set, sample_len):
    acc = 0
    predictions = []
    for char_seq, label in data_set:
        pos_score = 0
        count = 0
        prediction_entry = {
            'label': label,
            'micro': [],
            'pred': 0,
            'score': 0.
        }
        predictions.append(prediction_entry)
        for c in range(0, len(char_seq), sample_len):
            str = char_seq[c: c + sample_len]
            sequence_in = autograd.Variable(torch.LongTensor([prepare_sequence(str, len(str))]))
            score = model(sequence_in)
            pos_score += score.data.numpy()[0][1]
            prediction_entry['micro'].append(score.data.numpy()[0][1])
            count += 1

        pos_score = pos_score / count
        pred = 1 if pos_score >= 0.5 else 0

        prediction_entry['pred'] = pred
        prediction_entry['score'] = pos_score

        acc += 1 if pred == label else 0

    acc = 100 * acc / len(data_set)
    return acc, predictions


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data', type=str, default="./data_obf")
    argparser.add_argument('--batch_test', type=bool, default=True)
    argparser.add_argument('--model_path', type=str, default="models_obf")
    argparser.add_argument('--sample_len', type=int, default=40)
    args = argparser.parse_args()

    dirs = [Path(args.data)]
    if args.batch_test:
        dirs = dirs[0].iterdir()

    accuracies = []
    for dir in dirs:

        model_path = dir
        if args.model_path is not None:
            model_path = Path(args.model_path)

        model_path = model_path / ("model_" + dir.name + ".pt")
        if not model_path.exists():
            continue

        print("\n\nLoading data from %s ..." % dir.name)
        test_dir = Path(dir / "test")
        test_data = read_dir(test_dir)

        model = torch.load(model_path)
        acc, predictions = test(model, test_data, args.sample_len)

        for p in predictions:
            print("Actual: %d \t Predicted: %d \t Score: %3.2f" % (p['label'], p['pred'], p['score']))

        accuracies.append(acc)

    acc = sum(accuracies)/len(accuracies)
    print("\nOverall Accuracy: %3.2f" % acc)
