import torch
import torch.nn as nn
from torch import optim
import torch.autograd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random


class BiLSTM(nn.Module):
    def __init__(self, embedding, embedding_size, hidden_size, n_layers, batch_size):
        super(BiLSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.embedded = nn.Embedding.from_pretrained(torch.FloatTensor(embedding))
        self.bilstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=n_layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size*2, 1, bias=True)  # positive or negative review
        self.hidden = self.init_hidden()
        self.LogSoftmax = nn.LogSoftmax(dim=0)

    def init_hidden(self):
        # TODO: verify that this is correct
        return (torch.autograd.Variable(torch.zeros(4, self.batch_size, self.hidden_size)),
                torch.autograd.Variable(torch.zeros(4, self.batch_size, self.hidden_size)))

    def forward(self, sent, lengths):
        x = self.embedded(sent.long())
        batch_size, seq_len, _ = x.size()

        # Sort sentences by length then pack it
        x, lengths = sort_seqs(x, lengths)
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths=lengths, batch_first=True)

        out, self.hidden = self.bilstm(x_packed, self.hidden)

        # Pad sequence
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out_padded = torch.zeros(batch_size, seq_len, 512)  # batch_size, max_sequence_len, hidden_size
        out_padded[:, 0:out.size()[1], :] = out
        out_padded = out_padded.contiguous()
        check_out = out_padded.view(-1, out_padded.shape[2])

        prediction = self.linear(check_out)
        log_probs = self.LogSoftmax(prediction)
        log_probs = log_probs.view(batch_size, seq_len)
        # TODO: verify the following:
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        return log_probs

    # def predict_forward(self, sent, length):
    #     x = self.embedded(torch.from_numpy(sent).long())
    #     # batch_size, seq_len, _ = x.size()
    #
    #     # Sort sentences by length then pack it
    #     # x, lengths = sort_seqs(x, lengths)
    #
    #     x_packed = x[0:length]
    #     x_packed = x_packed.unsqueeze(0)
    #     out, self.hidden = self.bilstm(x_packed, self.hidden)
    #
    #     # Pad sequence
    #     out_padded = torch.zeros(60, 512)  # max_sequence_len, hidden_size
    #     out_padded[:, 0:out.size()[1]] = out
    #     out_padded = out_padded.contiguous()
    #     check_out = out_padded.view(-1, out_padded.shape[2])
    #
    #     prediction = self.linear(check_out)
    #     log_probs = self.LogSoftmax(prediction)
    #     log_probs = log_probs.view(60)
    #     return log_probs


def sort_seqs(embeddings, lengths):
    sorted_lengths, permutation_index = lengths.sort(0, descending=True)
    sorted_tensor = embeddings.index_select(0, permutation_index)
    return sorted_tensor, sorted_lengths


def load_data(train_x, train_y, train_seq_lens, batch_size):
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(train_seq_lens))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader


def bilstm_training(batch_size, train_labels, train_seq_lens, train_mat, bilstm):
    loader = load_data(train_mat, train_labels, train_seq_lens, batch_size)

    optimizer = optim.Adam(bilstm.parameters(), lr=1e-3)
    loss_func = nn.NLLLoss()
    bilstm.train()
    for epoch in range(0, 1):
        # batching below
        total_loss = 0.0
        i = 0
        for sentences, labels, seq_lengths in loader:
            # print(sentences)
            # print(labels)
            # print(seq_lengths)
            bilstm.init_hidden()
            bilstm.zero_grad()
            probs = bilstm.forward(sentences, seq_lengths)
            loss = loss_func(probs, labels.long())
            total_loss += loss
            loss.backward(retain_graph=True)
            optimizer.step()
            i += 1
            print(i)
        print("Loss on epoch %i: %f" % (epoch, total_loss))
    return
