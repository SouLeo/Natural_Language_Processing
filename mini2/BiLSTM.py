import torch
import torch.nn as nn
from torch import optim
import torch.autograd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import time

# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py

class BiLSTM(nn.Module):
    def __init__(self, embedding, embedding_size, hidden_size, n_layers, batch_size):
        super(BiLSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedded = nn.Embedding.from_pretrained(torch.FloatTensor(embedding))
        self.bilstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=n_layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size*n_layers*2, 2, bias=True)  # positive or negative review
        self.hidden = self.init_hidden()
        # self.LogSoftmax = nn.LogSoftmax(dim=0)

    def init_hidden(self):
        return (torch.zeros(self.n_layers*2, self.batch_size, self.hidden_size),
                torch.zeros(self.n_layers*2, self.batch_size, self.hidden_size))

    def forward(self, sent, lengths):
        self.hidden = self.init_hidden()
        x = self.embedded(sent.long())
        # batch_size, seq_len, _ = x.size()

        # Sort sentences by length then pack it
        # x, lengths, perm_indx = sort_seqs(x, lengths)
        # x_packed_sorted = torch.nn.utils.rnn.pack_padded_sequence(x, lengths=lengths, batch_first=True)

        out, _ = self.bilstm(x, self.hidden)

        # Pad sequence
        # out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # out: batch_size x padded_seq_len x hidden_size*2

        # _, orig_idx = perm_indx.sort(0)
        # unsorted_out = out[orig_idx]


        hi = out[:, -1, :]  # batch_size x hidden_size*2
        prediction = self.linear(hi)  # batch_size x num_tags=2

        # log_probs = self.LogSoftmax(prediction)  # batch_size x num_tags=2
        return prediction


def sort_seqs(embeddings, lengths):
    sorted_lengths, permutation_index = lengths.sort(0, descending=True)
    sorted_tensor = embeddings.index_select(0, permutation_index)
    return sorted_tensor, sorted_lengths, permutation_index


def load_data(train_x, train_y, train_seq_lens, batch_size):
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(train_seq_lens))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader


def bilstm_training(batch_size, train_labels, train_seq_lens, train_mat, bilstm):
    loader = load_data(train_mat, train_labels, train_seq_lens, batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(bilstm.parameters(), lr=3e-3)

    # bilstm.train()
    for epoch in range(0, 10):
        # batching below
        total_loss = 0.0
        i = 0
        start_time = time.time()
        for sentences, labels, seq_lengths in loader:
            probs = bilstm.forward(sentences, seq_lengths)
            loss = criterion(probs, labels.long())
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1
            print(i)
        end_time = time.time()
        print('epoch time duration:')
        print(end_time-start_time)
        print("Loss on epoch %i: %f" % (epoch, total_loss))
    return
