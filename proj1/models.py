# models.py

from optimizers import *
from nerdata import *
from utils import *

from collections import Counter
from typing import List

import numpy as np


class ProbabilisticSequenceScorer(object):
    """
    Scoring function for sequence models based on conditional probabilities.
    Scores are provided for three potentials in the model: initial scores (applied to the first tag),
    emissions, and transitions. Note that CRFs typically don't use potentials of the first type.

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """

    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs: np.ndarray,
                 transition_log_probs: np.ndarray, emission_log_probs: np.ndarray):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence_tokens: List[Token], tag_idx: int):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence_tokens: List[Token], prev_tag_idx: int, curr_tag_idx: int):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence_tokens: List[Token], tag_idx: int, word_posn: int):
        word = sentence_tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of(
            "UNK")
        return self.emission_log_probs[tag_idx, word_idx]


class FeatureBasedSequenceScorer(object):
    """
    Scoring function for sequence models based on conditional probabilities.
    Scores are provided for three potentials in the model: initial scores (applied to the first tag),
    emissions, and transitions. Note that CRFs typically don't use potentials of the first type.

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """

    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, emission_potentials: np.ndarray):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.emission_potentials = emission_potentials

    # TODO: finish converting this to emission feature based scoring. remove references to 'word'
    def score_emission(self, sentence_tokens: List[Token], tag_idx: int, word_posn: int):
        word = sentence_tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of(
            "UNK")
        return self.emission_potentials[tag_idx, word_idx]


class HmmNerModel(object):
    """
    HMM NER model for predicting tags

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """

    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs, transition_log_probs,
                 emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        # self.final_log_probs = final_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def decode(self, sentence_tokens: List[Token]):
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """

        tag_seq_indx = self.viterbi_algorithm(sentence_tokens)
        tag_labels = []
        for i in range(len(sentence_tokens)):
            indx = int(tag_seq_indx[i])
            tag_label = self.tag_indexer.get_object(indx)
            tag_labels.append(tag_label)
        chunks = chunks_from_bio_tag_seq(tag_labels)
        return LabeledSentence(sentence_tokens, chunks)

    def viterbi_algorithm(self, sentence_tokens: List[Token]):
        pss = ProbabilisticSequenceScorer(self.tag_indexer, self.word_indexer, self.init_log_probs,
                                          self.transition_log_probs, self.emission_log_probs)

        sent_len = len(sentence_tokens)
        trans_mat = self.transition_log_probs

        v = np.zeros((sent_len, trans_mat.shape[0]))
        best = np.zeros((sent_len, trans_mat.shape[0]))

        # Handle the initial state
        for y in range(trans_mat.shape[0]):
            v[0, y] = pss.score_init(sentence_tokens, y) + pss.score_emission(sentence_tokens, y, 0)
            best[0, y] = 0

        for i in range(1, sent_len):
            for y in range(trans_mat.shape[0]):
                v[i, y] = np.max(pss.score_emission(sentence_tokens, y, i) + trans_mat[:, y] + v[i - 1, :])
                best[i, y] = np.argmax(trans_mat[:, y] + v[i - 1, :])

        x = np.zeros(sent_len)
        x[-1] = np.argmax(v[sent_len - 1, :])

        for j in range(sent_len - 1, 0, -1):
            x[j - 1] = best[j, int(x[j])]
        return x


def train_hmm_model(sentences: List[LabeledSentence]) -> HmmNerModel:
    """
    Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
    Any word that only appears once in the corpus is replaced with UNK. A small amount
    of additive smoothing is applied.
    :param sentences: training corpus of LabeledSentence objects
    :return: trained HmmNerModel
    """
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.add_and_get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter[token.word] += 1.0
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    # final_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer), len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer), len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in range(0, len(sentence)):
            tag_idx = tag_indexer.add_and_get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_idx] += 1.0
            else:
                # if i == (len(sentence) - 1):
                #     final_counts[tag_idx] += 1.0
                transition_counts[tag_indexer.add_and_get_index(bio_tags[i - 1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    print(repr(init_counts))
    init_counts = np.log(init_counts / init_counts.sum())

    # print(repr(final_counts))
    # final_counts = np.log(final_counts / final_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    print("Tag indexer: %s" % tag_indexer)
    print("Initial state log probabilities: %s" % init_counts)
    print("Transition log probabilities: %s" % transition_counts)
    print("Emission log probs too big to print...")
    print("Emission log probs for India: %s" % emission_counts[:, word_indexer.add_and_get_index("India")])
    print("Emission log probs for Phil: %s" % emission_counts[:, word_indexer.add_and_get_index("Phil")])
    print("   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)")
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


def get_word_index(word_indexer: Indexer, word_counter: Counter, word: str) -> int:
    """
    Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
    At test time, unknown words will be replaced by UNKs.
    :param word_indexer: Indexer mapping words to indices for HMM featurization
    :param word_counter: Counter containing word counts of training set
    :param word: string word
    :return: int of the word index
    """
    if word_counter[word] < 1.5:
        return word_indexer.add_and_get_index("UNK")
    else:
        return word_indexer.add_and_get_index(word)


class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights, emission_potentials):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        self.emission_potentials = emission_potentials

    def decode(self, sentence_tokens):
        print('hi')
        tag_seq_indx = self.compute_forward_backward(sentence_tokens)
        tag_labels = []
        for i in range(len(sentence_tokens)):
            indx = int(tag_seq_indx[i])
            tag_label = self.tag_indexer.get_object(indx)
            tag_labels.append(tag_label)
        chunks = chunks_from_bio_tag_seq(tag_labels)
        return LabeledSentence(sentence_tokens, chunks)


def compute_forward_backward(emission_potentials, sentence_tokens: List[Token], tag_indexer):
    # Everything is in logspace
    # fbss = ProbabilisticSequenceScorer(tag_indexer, word_indexer, emission_potentials)

    sent_len = len(sentence_tokens)
    num_labels = len(tag_indexer)

    a = np.zeros((num_labels, sent_len))
    b = np.zeros((num_labels, sent_len))

    # Handle the initial state of a and b
    for i in range(num_labels):
        a[i][0] = emission_potentials[i][0]
        b[i][sent_len-1] = np.log(1)

    for i in range(num_labels):  # for all NER tags
        for t in range(1, sent_len):  # for all words in the sentence
            # forward pass
            for label_ind in range(num_labels):
                a[i][t] += np.exp(a[label_ind][t - 1] + emission_potentials[i][t])
            a[i][t] = np.log(a[i][t])

        for t in range(0, sent_len-1):
            # backward pass
            for label_ind in range(num_labels):
                b[i][t] += np.exp(b[label_ind][t + 1] + emission_potentials[i][t + 1])
            b[i][t] = np.log(b[i][t])

    # TODO: use a and b to build marginals
    marginals = np.zeros((num_labels, sent_len))
    for i in range(num_labels):
        for t in range(1, sent_len):
            # TODO: Solve why sum is not the same for all iterations
            sum = np.sum(np.multiply(a[:, t], b[:, t]))
            marginals[i][t] = (a[i][t]*b[i][t])/sum
    print('fb ended')
    return marginals


# Trains a CrfNerModel on the given corpus of sentences.
def train_crf_model(sentences: List[LabeledSentence]) -> CrfNerModel:
    tag_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    print("Extracting features")
    feature_indexer = Indexer()
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in range(0, len(tag_indexer))] for j in range(0, len(sentences[i]))] for i in
                     range(0, len(sentences))]
    for sentence_idx in range(0, len(sentences)):
        if sentence_idx % 100 == 0:
            print("Ex %i/%i" % (sentence_idx, len(sentences)))
        for word_idx in range(0, len(sentences[sentence_idx])):
            for tag_idx in range(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(
                    sentences[sentence_idx].tokens, word_idx, tag_indexer.get_object(tag_idx), feature_indexer,
                    add_to_indexer=True)

    print("Emission Training")
    feat_len = len(feature_indexer.ints_to_objs)  # * 8  # only includes emission features
    feature_weights = np.zeros(feat_len)
    learning_rate = 0.5
    epochs = 3
    sgd = SGDOptimizer(feature_weights, learning_rate)

    for epoch in range(epochs):
        for sentence_idx in range(0, len(sentences)):
            # TODO: Implement emission potential update [DONE?]
            emission_potentials = np.zeros((len(tag_indexer), len(sentences[sentence_idx])))
            for word_idx in range(0, len(sentences[sentence_idx])):
                for tag_idx in range(0, len(tag_indexer)):
                    emission_potentials[tag_idx][word_idx] = score_indexed_features(
                        feature_cache[sentence_idx][word_idx][tag_idx], sgd.weights)
            # TODO: Implement forward-backward for marginals for emission
            # print('hi')
            compute_forward_backward(emission_potentials, sentences[sentence_idx], tag_indexer)
            # TODO: Compute Grad over all emission probabilities

    return  # CrfNerModel(tag_indexer, feature_indexer, feature_weights, emission_potentials)


def extract_emission_features(sentence_tokens: List[Token], word_index: int, tag: str, feature_indexer: Indexer,
                              add_to_indexer: bool):
    """
    Extracts emission features for tagging the word at word_index with tag.
    :param sentence_tokens: sentence to extract over
    :param word_index: word index to consider
    :param tag: the tag that we're featurizing for
    :param feature_indexer: Indexer over features
    :param add_to_indexer: boolean variable indicating whether we should be expanding the indexer or not. This should
    be True at train time (since we want to learn weights for all features) and False at test time (to avoid creating
    any features we don't have weights for).
    :return: an ndarray
    """
    feats = []
    curr_word = sentence_tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in range(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_word = "</s>"
        else:
            active_word = sentence_tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_pos = "</S>"
        else:
            active_pos = sentence_tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in range(1, max_ngram_size + 1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in range(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    return np.asarray(feats, dtype=int)
