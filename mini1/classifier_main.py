# classifier_main.py

import argparse
import sys
import time
from nerdata import *
from utils import *
from collections import Counter
from optimizers import *
from typing import List
import numpy as np


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='BAD', help='model to run (BAD, CLASSIFIER)')
    parser.add_argument('--train_path', type=str, default='data/eng.train',
                        help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/eng.testa',
                        help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default='data/eng.testb.blind',
                        help='path to dev set (you should not need to modify)')
    parser.add_argument('--test_output_path', type=str, default='eng.testb.out',
                        help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false',
                        help='skip printing output on the test set')
    args = parser.parse_args()
    return args


class PersonExample(object):
    """
    Data wrapper for a single sentence for person classification, which consists of many individual tokens to classify.

    Attributes:
        tokens: the sentence to classify
        labels: 0 if non-person name, 1 if person name for each token in the sentence
    """

    def __init__(self, tokens: List[str], labels: List[int], pos: List[str]):
        self.tokens = tokens
        self.labels = labels
        self.pos = pos

    def __len__(self):
        return len(self.tokens)


def transform_for_classification(ner_exs: List[LabeledSentence]):
    """
    :param ner_exs: List of chunk-style NER examples
    :return: A list of PersonExamples extracted from the NER data
    """
    for labeled_sent in ner_exs:
        tags = bio_tags_from_chunks(labeled_sent.chunks, len(labeled_sent))
        pos = [tok.pos for tok in labeled_sent.tokens]
        labels = [1 if tag.endswith("PER") else 0 for tag in tags]
        # print(labels)
        yield PersonExample([tok.word for tok in labeled_sent.tokens], labels, pos)


class CountBasedPersonClassifier(object):
    """
    Person classifier that takes counts of how often a word was observed to be the positive and negative class
    in training, and classifies as positive any tokens which are observed to be positive more than negative.
    Unknown tokens or ties default to negative.
    Attributes:
        pos_counts: how often each token occurred with the label 1 in training
        neg_counts: how often each token occurred with the label 0 in training
    """

    def __init__(self, pos_counts: Counter, neg_counts: Counter):
        self.pos_counts = pos_counts
        self.neg_counts = neg_counts

    def predict(self, tokens: List[str], idx: int):
        if self.pos_counts[tokens[idx]] > self.neg_counts[tokens[idx]]:
            return 1
        else:
            return 0


def train_count_based_binary_classifier(ner_exs: List[PersonExample]):
    """
    :param ner_exs: training examples to build the count-based classifier from
    :return: A CountBasedPersonClassifier using counts collected from the given examples
    """
    # print("Labels Print:")
    # print(ner_exs[1].labels)
    # print("Tokens Print:")
    # print(ner_exs[1].tokens)
    # Determine Vocab Size from len of all Tokens in list

    pos_counts = Counter()
    neg_counts = Counter()
    for ex in ner_exs:
        for idx in range(0, len(ex)):
            if ex.labels[idx] == 1:
                pos_counts[ex.tokens[idx]] += 1.0
            else:
                neg_counts[ex.tokens[idx]] += 1.0
    print(repr(pos_counts))
    print(repr(pos_counts["Peter"]))
    print(repr(pos_counts["aslkdjtalk;sdjtakl"]))
    return CountBasedPersonClassifier(pos_counts, neg_counts)


class PersonClassifier(object):
    """
    Classifier to classify a token in a sentence as a PERSON token or not.
    Constructor arguments are merely suggestions; you're free to change these.
    """

    def __init__(self, weights: np.ndarray, indexer: Indexer):
        self.weights = weights
        self.indexer = indexer

    def predict(self, tokens: List[str], idx: int):
        """
        Makes a prediction for token at position idx in the given PersonExample
        :param tokens:
        :param idx:
        :return: 0 if not a person token, 1 if a person token
        """
        # print('hi')

    def predict(self, ex, idx):
        # TODO: Create feature vector for word
        # ex.tokens.insert(0, "<s>")
        # ex.pos.insert(0, "<s>")
        # idx = idx + 1 # because I added a start token at the beginning of each sentence
        curr_word = ex.tokens[idx]
        if curr_word == '<s>':
            return 0
        curr_word_indx = self.indexer.index_of(ex.tokens[idx])
        prev_word_indx = self.indexer.index_of(ex.tokens[idx - 1])

        # Step 3: Convert ints to One Hot Encoded Vectors
        curr_word_onehot_encoding = int_index_to_one_hot_vector(curr_word_indx, len(self.indexer))
        prev_word_onehot_encoding = int_index_to_one_hot_vector(prev_word_indx, len(self.indexer))
        # Step 4: Add additional features
        init_cap_encode = init_cap_encoding(ex.tokens[idx])
        # TODO: FIX BOTTOM LINE
        pos_encode = pos_encoding(ex.pos[idx])
        # print(init_cap_encode)
        # print(pos_encode)
        # Step 5: Concatenate to form bigram
        feature_vec = prev_word_onehot_encoding + curr_word_onehot_encoding + init_cap_encode + pos_encode
        # feature_vec = curr_word_onehot_encoding
        # indices = np.nonzero(np.asarray(bigram))
        # bigram_counter = Counter(list(indices[0].astype(int)))
        # for i in bigram_counter.keys():
        #     bigram_counter[i] = bigram[i]
        # feature_vec = bigram_counter
        # Prediction:
        prediction = np.dot(self.weights.transpose(), feature_vec)
        if prediction >= 0:
            label = 1
        else:
            label = 0
        return label


def pos_encoding(tag):
    pos_tags = [
        "<s>", "-X-", ":", ".", ",", "(", ")", "$", '\"', "\'\'",
        "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS",
        "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP",
        "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB",
        "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB",
        "NN|SYM"
    ]
    pos_to_int = dict((pos, i) for i, pos in enumerate(pos_tags))
    # int_to_pos = dict((i, pos) for i, pos in enumerate(pos_tags))

    # print(tag)
    i = pos_to_int[tag]
    pos_onehot_encoded = [0 for _ in range(len(pos_tags))]
    pos_onehot_encoded[i] = 1
    # print(pos_onehot_encoded)
    return pos_onehot_encoded


def init_cap_encoding(token):
    init_cap = []
    if token.isupper():
        init_cap.append(1)
        init_cap.append(0)
    else:
        init_cap.append(0)
        init_cap.append(1)
    # print(init_cap)
    return init_cap


def generate_unique_vocabulary(ner_exs: List[PersonExample]):
    # Step 1: Find List of Unique Tokens and
    #         Use Their Indx (int) for Unique Feature ID
    vocab = Indexer()
    for ex in ner_exs:
        for idx in range(0, len(ex)):
            vocab.add_and_get_index(ex.tokens[idx])
    # adds a string start token to the vocabulary
    vocab.add_and_get_index("<s>")
    # print(len(vocab))
    # print(vocab.ints_to_objs)
    return vocab


def int_index_to_one_hot_vector(indx: int, vocab_len: int):
    token = [0 for _ in range(vocab_len)]
    token[indx] = 1
    # print(len(token))
    return token


def create_bigram_model(ner_exs: List[PersonExample], vocab: Indexer):
    bigram_model = []
    labels = []
    # print(len(ner_exs))
    for ex in ner_exs:
        # Step 1: Force each sentence to begin with a start token
        ex.tokens.insert(0, "<s>")
        ex.pos.insert(0, "<s>")
        ex.labels.insert(0, 0)
        # associate PoS tag with <s> rn
        for idx in range(1, len(ex)):
            # Get token label
            labels.append(ex.labels[idx])
            # Step 2: Find index of current word and previous word
            curr_word_indx = vocab.index_of(ex.tokens[idx])
            prev_word_indx = vocab.index_of(ex.tokens[idx-1])
            curr_word = ex.tokens[idx]
            prev_word = ex.tokens[idx-1]
            # Step 3: Convert ints to One Hot Encoded Vectors
            curr_word_onehot_encoding = int_index_to_one_hot_vector(curr_word_indx, len(vocab))
            prev_word_onehot_encoding = int_index_to_one_hot_vector(prev_word_indx, len(vocab))
            # Step 4: Add additional features
            init_cap_encode = init_cap_encoding(ex.tokens[idx])
            pos_encode = pos_encoding(ex.pos[idx])
            # print(init_cap_encode)
            # print(pos_encode)
            # Step 5: Concatenate to form bigram
            bigram = prev_word_onehot_encoding + curr_word_onehot_encoding + init_cap_encode + pos_encode
            # bigram = curr_word_onehot_encoding
            bigram_feat_len = len(bigram)

            indices = np.nonzero(np.asarray(bigram))
            bigram_counter = Counter(list(indices[0].astype(int)))

            for i in bigram_counter.keys():
                bigram_counter[i] = bigram[i]
            # Step 6: Add feature vector to bigram_model
            # print(len(bigram))
            bigram_model.append(bigram_counter)

    return bigram_model, bigram_feat_len, labels


def train_classifier(ner_exs: List[PersonExample]):
    vocabulary = generate_unique_vocabulary(ner_exs)
    bigram_model, feature_length, labels = create_bigram_model(ner_exs, vocabulary)

    weights = np.zeros(feature_length)
    d_weights = Counter()  # np.zeros(feature_len)
    learning_rate = 0.1
    epochs = 3
    print('entering training loop')
    sgd_algo = SGDOptimizer(weights, learning_rate)
    for epoch in range(epochs):
        print(epoch)
        for x in range(len(bigram_model)):
            # Create one hot vector for each training example
            x_onehot = np.zeros(feature_length)
            for i in bigram_model[x].keys():
                x_onehot[i] = bigram_model[x][i]
            # For each nonzero feature value in a given training example,
            # calculate the gradient update
            for j in bigram_model[x].keys():
                d_weights[j] = bigram_model[x][j]*(labels[x]-(1/(1+np.exp(np.dot(-sgd_algo.weights.transpose(),
                                                                                 np.asarray(x_onehot))))))
            sgd_algo.apply_gradient_update(d_weights, batch_size=1)
    print(d_weights)
    classifier = PersonClassifier(sgd_algo.get_final_weights(), vocabulary)
    return classifier

def evaluate_classifier(exs: List[PersonExample], classifier: PersonClassifier):
    """
    Prints evaluation of the classifier on the given examples
    :param exs: PersonExample instances to run on
    :param classifier: classifier to evaluate
    """
    predictions = []
    golds = []
    for ex in exs:
        for idx in range(0, len(ex)):
            golds.append(ex.labels[idx])
            # predictions.append(classifier.predict(ex.tokens, idx))
            predictions.append(classifier.predict(ex, idx))
    print_evaluation(golds, predictions)


def print_evaluation(golds: List[int], predictions: List[int]):
    """
    Prints statistics about accuracy, precision, recall, and F1
    :param golds: list of {0, 1}-valued ground-truth labels for each token in the test set
    :param predictions: list of {0, 1}-valued predictions for each token
    :return:
    """
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    if len(golds) != len(predictions):
        raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
    for idx in range(0, len(golds)):
        gold = golds[idx]
        prediction = predictions[idx]
        if prediction == gold:
            num_correct += 1
        if prediction == 1:
            num_pred += 1
        if gold == 1:
            num_gold += 1
        if prediction == 1 and gold == 1:
            num_pos_correct += 1
        num_total += 1
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
    prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
    rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0
    print("Precision: %i / %i = %f" % (num_pos_correct, num_pred, prec))
    print("Recall: %i / %i = %f" % (num_pos_correct, num_gold, rec))
    print("F1: %f" % f1)


def predict_write_output_to_file(exs: List[PersonExample], classifier: PersonClassifier, outfile: str):
    """
    Runs prediction on exs and writes the outputs to outfile, one token per line
    :param exs:
    :param classifier:
    :param outfile:
    :return:
    """
    f = open(outfile, 'w')
    for ex in exs:
        for idx in range(0, len(ex)):
            prediction = classifier.predict(ex.tokens, idx)
            f.write(ex.tokens[idx] + " " + repr(int(prediction)) + "\n")
        f.write("\n")
    f.close()


if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()
    print(args)
    # Load the training and test data
    train_class_exs = list(transform_for_classification(read_data(args.train_path)))
    dev_class_exs = list(transform_for_classification(read_data(args.dev_path)))
    training_truncation = 3000
    testing_truncation = int(training_truncation/2)

    # Train the model
    if args.model == "BAD":
        classifier = train_count_based_binary_classifier(train_class_exs)
    else:
        classifier = train_classifier(train_class_exs[1:training_truncation])
    print("Data reading and training took %f seconds" % (time.time() - start_time))
    # Evaluate on training, development, and test data
    # print("===Train accuracy===")
    # evaluate_classifier(train_class_exs[1:training_truncation], classifier)
    print("===Dev accuracy===")
    evaluate_classifier(dev_class_exs[1:testing_truncation], classifier)
    # if args.run_on_test:
    #     print("Running on test")
    #     test_exs = list(transform_for_classification(read_data(args.blind_test_path)))
    #     predict_write_output_to_file(test_exs, classifier, args.test_output_path)
    #     print("Wrote predictions on %i labeled sentences to %s" % (len(test_exs), args.test_output_path))