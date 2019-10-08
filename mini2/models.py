# models.py

from sentiment_data import *
from typing import List
from FFNN import *

def pad_to_length(np_arr, length):
    """
    Forces np_arr to length by either truncation (if longer) or zero-padding (if shorter)
    :param np_arr:
    :param length: Length to pad to
    :return: a new numpy array with the data from np_arr padded to be of length length. If length is less than the
    length of the base array, truncates instead.
    """
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result


# , using dev_exs for development and returning
# predictions on the *blind* test_exs (all test_exs have label 0 as a dummy placeholder value).
def train_evaluate_ffnn(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], test_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> List[SentimentExample]:
    """
    Train a feedforward neural network on the given training examples, using dev_exs for development, and returns
    predictions on the *blind* test examples passed in. Returned predictions should be SentimentExample objects with
    predicted labels and the same sentences as input (but these won't be read by the external code). The code is set
    up to go all the way to test predictions so you have more freedom to handle example processing as you see fit.
    :param train_exs:
    :param dev_exs:
    :param test_exs:
    :param word_vectors:
    :return:
    """
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])

    input_size = word_vectors.get_embedding_length()
    hid_size = word_vectors.get_embedding_length()
    num_classes = 2
    ffnn = FFNN(input_size, hid_size, num_classes)  # TODO: Add embed layer?
    lr = 0.001
    epochs = 5

    learn_weights(lr, epochs, ffnn, train_labels_arr, train_mat, word_vectors, num_classes, train_seq_lens)

    # Begin Prediction
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    guesses = []
    correct = 0
    for idx in range(0, len(dev_exs)):
        # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
        # quantities from the running of the computation graph, namely the probabilities, prediction, and z
        x = form_input(dev_mat[idx], word_vectors, dev_seq_lens[idx])
        probs = ffnn.forward(x)
        prediction = int(torch.argmax(probs))
        guess = SentimentExample(dev_exs[idx].indexed_words, prediction)
        guesses.append(guess)
        if prediction == dev_exs[idx].label:
            correct += 1
    accuracy = 100*correct/len(dev_exs)
    print('accuracy: ')
    print(accuracy)
    return guesses
    # raise Exception("Not implemented")


# Analogous to train_ffnn, but trains your fancier model.
def train_evaluate_fancy(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], test_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> List[SentimentExample]:
    # Create LSTM
    raise Exception("Not implemented")