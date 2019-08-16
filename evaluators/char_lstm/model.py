import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class AuthorLSTM(nn.Module):
    """
    The training examples are in the form of <a sequence of characters, label>.
    In which the label is a binary variable determining that the sequence is written
    by the author or not.
    This model maps a sequence of characters to an embedding space then
    builds an LSTM network on top of it. Finally a sigmoid function is used
    to convert the LSTM outputs to the probability of the author, given the sequence.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_size, n_classes, bi=True, n_layers=1):
        super(AuthorLSTM, self).__init__()

        self.embed_dim = embedding_dim
        self.hidden_dim = hidden_size
        self.out_dim = n_classes
        self.n_layers = n_layers
        self.bi = bi
        self.n_dir = 2 if bi else 1

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, n_layers, bidirectional=bi, dropout=True)
        self.linear = nn.Linear(self.hidden_dim * self.n_dir, n_classes)
        self.o = F.softmax

    def init_hidden(self, batch_size):
        c0 = autograd.Variable(torch.zeros(self.n_layers * self.n_dir, batch_size, self.hidden_dim))
        h0 = autograd.Variable(torch.zeros(self.n_layers * self.n_dir, batch_size, self.hidden_dim))
        return h0, c0

    def forward(self, inputs):
        batch_size = inputs.size(0)  # batches * Sequences
        inputs = inputs.t()  # Sequences * batches
        sequence_len = inputs.size(0)

        hidden = self.init_hidden(batch_size)
        embeds = self.embedding(inputs)
        out, hidden = self.rnn(embeds, hidden)
        last_layer = out[-1]
        linear = self.linear(last_layer)  # take the last layer
        score = self.o(linear.view(batch_size, -1), 1)
        return score
