# feedforward_example_pytorch.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random

# DEFINING THE COMPUTATION GRAPH
# Define the core neural network: one hidden layer, tanh nonlinearity
# Returns probabilities; in general your network can be set up to return probabilities, log probabilities,
# or (log) probabilities + loss
class FFNN(nn.Module):
    def __init__(self, inp, hid, out):
        super(FFNN, self).__init__()

        self.V = nn.Linear(inp, hid)
        self.g = nn.Tanh()
        self.W = nn.Linear(hid, out)
        self.LogSoftmax = nn.LogSoftmax(dim=0)
        # Initialize weights according to the Xavier Glorot formula
        nn.init.xavier_uniform(self.V.weight)
        nn.init.xavier_uniform(self.W.weight)

    # Forward computation. Backward computation is done implicitly (nn.Module already has an implementation of
    # it that you shouldn't need to override)
    def forward(self, x):
        # p = self.g(self.V(x))
        # u = torch.mean(p, 0)
        return self.LogSoftmax(self.W(self.g(self.V(x))))


# Form the input to the neural network. In general this may be a complex function that synthesizes multiple pieces
# of data, does some computation, handles batching, etc.
def form_input(sent, word_embedding, sent_len):
    embedded_input = np.zeros((word_embedding.get_embedding_length(), len(sent)))
    if len(sent) == 0:
        print('zero len!')
    for token_idx in range(len(sent)):
        word = word_embedding.word_indexer.get_object(int(sent[token_idx]))
        embedded_input[:, token_idx] = word_embedding.get_embedding(word)
    inp = torch.from_numpy(embedded_input).float()
    u = torch.sum(inp, 1)/len(sent)
    return u


def learn_weights(alpha, epochs, ffnn, train_labels, train_mat, word_vectors, num_classes, train_seq_lens):
    optimizer = optim.Adam(ffnn.parameters(), alpha)
    ffnn.train()
    loss_func = nn.NLLLoss()
    for epoch in range(epochs):
        train_lab_ind = [i for i in range(0, len(train_labels))]
        random.shuffle(train_lab_ind)
        total_loss = 0.0
        for idx in train_lab_ind:
            x = form_input(train_mat[idx], word_vectors, train_seq_lens[idx])
            y = train_labels[idx]
            # Build one-hot representation of y
            y_onehot = torch.zeros(num_classes)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y, dtype=np.int64)), 1)
            # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
            ffnn.zero_grad()
            probs = ffnn.forward(x)
            # Can also use built-in NLLLoss as a shortcut here (takes log probabilities) but we're being explicit here
            # print(y)
            y_tens = torch.LongTensor([int(y)])
            # print(y_tens)
            loss = loss_func(probs.unsqueeze(0), y_tens)

            # loss = torch.neg(torch.log(probs)).dot(y_onehot)
            total_loss += loss
            loss.backward()
            optimizer.step()
        print("Loss on epoch %i: %f" % (epoch, total_loss))
    return
