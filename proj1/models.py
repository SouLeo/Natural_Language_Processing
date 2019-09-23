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

    def __init__(self, tag_indexer: Indexer, feature_indexer: Indexer, feature_weights: np.ndarray):
        self.tag_indexer = tag_indexer
        # self.word_indexer = word_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights

    def score_init(self, feature_cache, tag_idx):
        return score_indexed_features(feature_cache[0][tag_idx], self.feature_weights)

    def score_transition(self, feature_cache, prev_tag_idx, curr_tag_idx):
        return 0

    def score_emission(self, feature_cache, tag_idx, word_idx):
        return score_indexed_features(feature_cache[word_idx][tag_idx], self.feature_weights)

    # # TODO: finish converting this to emission feature based scoring. remove references to 'word'
    # def score_emission(self, tag_idx: int, word_idx: int, feature_cache):
    #     # word = sentence_tokens[word_posn].word
    #     # word_idx = self.feature_indexer.index_of(word) if self.feature_indexer.contains(word) \
    #     #     else self.feature_indexer.index_of("UNK")
    #     return score_indexed_features(feature_cache[word_idx][tag_idx], self.feature_weights)


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
    def __init__(self, tag_indexer, feature_indexer, feature_weights):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights

    def decode(self, sentence):
        feature_cache = [[[] for k in range(0, len(self.tag_indexer))] for j in range(0, len(sentence))]
        for word_idx in range(0, len(sentence)):
            for tag_idx in range(0, len(self.tag_indexer)):
                feature_cache[word_idx][tag_idx] = extract_emission_features(sentence, word_idx,
                                                                             self.tag_indexer.get_object(tag_idx),
                                                                             self.feature_indexer, add_to_indexer=False)

        # Viterbi
        score = np.zeros((len(sentence), len(self.tag_indexer)))
        back_pointers = np.ones((len(sentence), len(self.tag_indexer))) * -1
        sequence_scorer = FeatureBasedSequenceScorer(self.tag_indexer, self.feature_indexer, self.feature_weights)
        for word_idx in range(0, len(sentence)):
            if word_idx == 0:
                for tag_idx in range(0, len(self.tag_indexer)):
                    tag = self.tag_indexer.get_object(tag_idx)
                    if isI(tag):
                        score[word_idx][tag_idx] = -np.inf
                    else:
                        score[word_idx][tag_idx] = sequence_scorer.score_init(feature_cache, tag_idx)
            else:
                for curr_tag_idx in range(0, len(self.tag_indexer)):
                    score[word_idx][curr_tag_idx] = -np.inf
                    for prev_tag_idx in range(0, len(self.tag_indexer)):
                        # TODO : did not prohibit the O-I transition at the last word
                        curr_tag = self.tag_indexer.get_object(curr_tag_idx)
                        prev_tag = self.tag_indexer.get_object(prev_tag_idx)
                        if isO(prev_tag) and isI(curr_tag):
                            continue
                        if isI(curr_tag) and (get_tag_label(curr_tag) != get_tag_label(prev_tag)):
                            continue
                        curr_score = sequence_scorer.score_transition(feature_cache, prev_tag_idx, curr_tag_idx) + \
                                     sequence_scorer.score_emission(feature_cache, curr_tag_idx, word_idx) + \
                                     score[word_idx - 1][prev_tag_idx]
                        if curr_score > score[word_idx][curr_tag_idx]:
                            score[word_idx][curr_tag_idx] = curr_score
                            back_pointers[word_idx][curr_tag_idx] = prev_tag_idx
        max_score_idx = score.argmax(axis=1)[-1]
        idx = max_score_idx
        pred_tags = []
        word_idx = len(sentence) - 1
        while idx != -1:
            pred_tags.append(self.tag_indexer.get_object(idx))
            idx = back_pointers[word_idx][int(idx)]
            word_idx -= 1
        pred_tags.reverse()
        return LabeledSentence(sentence, chunks_from_bio_tag_seq(pred_tags))


def compute_forward_backward(sentence_tokens, tag_indexer, feature_cache, feature_weights):
    # Everything is in logspace

    sent_len = len(sentence_tokens)
    num_labels = len(tag_indexer)

    log_a = np.zeros((sent_len, num_labels))
    log_b = np.zeros((sent_len, num_labels))

    # Handle the initial state of a and b
    for i in range(num_labels):
        log_a[0][i] = score_indexed_features(feature_cache[0][i], feature_weights)
        log_b[sent_len - 1][i] = np.log(1)  # TODO: Verify log(1) is right

    # forward pass
    for t in range(1, sent_len):  # for all words in the sentence
        for i in range(0, num_labels):  # for the current word
            log_a[t][i] = -np.inf
            for j in range(num_labels):  # for the previous word
                curr_tag = tag_indexer.get_object(i)
                prev_tag = tag_indexer.get_object(j)
                if isI(curr_tag) and get_tag_label(curr_tag) != get_tag_label(prev_tag):
                    continue
                log_a[t][i] = np.logaddexp(log_a[t][i], log_a[t - 1][j] +
                                           score_indexed_features(feature_cache[t][i], feature_weights))

    # backward pass
    for t in range(sent_len - 2, -1, -1):
        for i in range(0, num_labels):  # for the current word
            log_b[t][i] = -np.inf
            for k in range(num_labels):  # for the next word
                curr_tag = tag_indexer.get_object(i)
                next_tag = tag_indexer.get_object(k)
                if isI(next_tag) and get_tag_label(curr_tag) != get_tag_label(next_tag):
                    continue
                log_b[t][i] = np.logaddexp(log_b[t][i], log_b[t + 1][k] +
                                           score_indexed_features(feature_cache[t][k], feature_weights))

    # Use a and b to build marginals
    log_marginals = log_a + log_b
    for t in range(sent_len):
        z = -np.inf
        for i in range(num_labels):
            z = np.logaddexp(z, log_marginals[t][i])
        log_marginals[t] -= z
    # print('fb ended')
    return log_marginals


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
    print("End of Extraction")

    print("Emission Training")
    feat_len = len(feature_indexer)  # * 8  # only includes emission features
    feature_weights = np.random.rand(feat_len)

    learning_rate = 0.5
    batch_size = 1
    epochs = 3

    sgd = SGDOptimizer(feature_weights, learning_rate)
    crf = CrfNerModel(tag_indexer, feature_indexer, feature_weights)


    for epoch in range(epochs):
        for sentence_idx in range(0, len(sentences)):
            # Calculate Emission Potentials
            marginals = compute_forward_backward(sentences[sentence_idx], tag_indexer,
                                                 feature_cache[sentence_idx], crf.feature_weights)
            gradient = Counter()
            # Apply Grad Update
            for word_idx in range(0, len(sentences[sentence_idx])):
                for tag_idx in range(0, len(tag_indexer)):
                    for obj in feature_cache[sentence_idx][word_idx][tag_idx]:
                        if gradient[obj] != 0:  # if it already exists in the counter
                            gradient[obj] -= np.exp(marginals[word_idx][tag_idx])
                        else:
                            gradient[obj] = -np.exp(marginals[word_idx][tag_idx])
                truth_label = sentences[sentence_idx].get_bio_tags()[word_idx]
                truth_idx = tag_indexer.index_of(truth_label)
                for obj in feature_cache[sentence_idx][word_idx][truth_idx]:
                    if gradient[obj] != 0:  # if it already exists in the counter
                        gradient[obj] += 1.0
                    else:
                        gradient[obj] = 1.0
            sgd.apply_gradient_update(gradient, batch_size)
            crf.feature_weights = sgd.get_final_weights()
            gradient = Counter()
    return crf


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
