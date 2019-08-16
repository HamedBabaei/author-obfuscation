import string
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import argparse
from pathlib import Path
from model import AuthorLSTM
from utils import *
import sys


def train(model, X, Y, batch_size, n_epochs, prefix=""):
    batches = batch(X, Y, batch_size)
    for epoch in range(n_epochs):
        total_loss = 0.
        correct = 0
        for b_x, b_y in batches:
            input = autograd.Variable(torch.LongTensor(b_x))
            target = autograd.Variable(torch.LongTensor(b_y), requires_grad=False)

            score = model(input)
            pred = score.data.numpy().argmax(1)
            correct += sum([1 if pred[i] == b_y[i] else 0 for i in range(0, len(b_y))])
            loss = loss_function(score, target)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss
            sys.stdout.write(".")
            sys.stdout.flush()

        print("\n%s \t Epoch %4d \t Loss: %.4f \t Accuracy: %3.2f" % (
        prefix, epoch + 1, total_loss, 100 * correct / len(X)))


def test(data_set):
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
        for c in range(0, len(char_seq), args.chunk_len):
            str = char_seq[c: c + args.chunk_len]
            sequence_in = autograd.Variable(torch.LongTensor([prepare_sequence(str, len(str))]))
            score = model(sequence_in)
            pos_score += score.data.numpy()[0][1]
            prediction_entry['micro'].append(score.data.numpy()[0][1])
            count += 1

        pos_score = pos_score / count
        pred = 1 if pos_score >= 0.5 else 0
        p = pos_score if pred == 1 else 1 - pos_score

        prediction_entry['pred'] = pred
        prediction_entry['score'] = p

        acc += 1 if pred == label else 0

    acc = 100 * acc / len(data_set)
    return acc, predictions


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data', type=str, default="./data_obf/")
    argparser.add_argument('--batch_train', type=bool, default=True)
    argparser.add_argument('--model_path', type=str, default="models_obf")
    argparser.add_argument('--sample_len', type=int, default=40)
    argparser.add_argument('--n_samples', type=int, default=2000)
    argparser.add_argument('--batch_size', type=int, default=500)
    argparser.add_argument('--epochs', type=int, default=20)
    argparser.add_argument('--hidden_size', type=int, default=100)
    argparser.add_argument('--learning_rate', type=float, default=0.01)
    args = argparser.parse_args()

    dirs = [Path(args.data)]
    if args.batch_train:
        dirs = dirs[0].iterdir()

    for dir in dirs:
        if dir.name.startswith("."):
            continue

        train_dir = Path(dir / "train")
        print("\nLoading data from %s ..." % dir.name)
        train_data = read_dir(train_dir)
        X_train, Y_train = get_random_samples(train_data, args.sample_len, args.n_samples)

        model = AuthorLSTM(len(char_to_ix), args.hidden_size, args.hidden_size, 2, False, 1)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        print("training started(N=%d) ..." % len(X_train))
        train(model, X_train, Y_train, args.batch_size, args.epochs, dir.name)

        model_path = dir
        if args.model_path is not None:
            model_path = Path(args.model_path)
            model_path.mkdir(parents=True, exist_ok=True)

        torch.save(model, model_path / ("model_" + dir.name + ".pt"))
