# print('hi')
        num_labels = len(self.tag_indexer)
        sent_len = len(sentence_tokens)
        # Extract Features of Input Sentence
        feature_cache = [[[] for k in range(0, num_labels)] for j in range(0, sent_len)]
        for word_idx in range(0, len(sentence_tokens)):
            for tag_idx in range(0, len(self.tag_indexer)):
                feature_cache[word_idx][tag_idx] = extract_emission_features(
                    sentence_tokens, word_idx, self.tag_indexer.get_object(tag_idx),
                    self.feature_indexer, add_to_indexer=False)

        fbss = FeatureBasedSequenceScorer(self.tag_indexer, self.feature_indexer, self.feature_weights)
        marginals = np.zeros((len(sentence_tokens), len(self.tag_indexer)))
        history = np.zeros((len(sentence_tokens), len(self.tag_indexer)))

        for word_idx in range(sent_len):
            if word_idx == 0:
                for tag_idx in range(num_labels):
                    tag = self.tag_indexer.get_object(tag_idx)
                    if isI(tag):
                        marginals[word_idx][tag_idx] = -np.inf
                    else:
                        marginals[word_idx][tag_idx] = fbss.score_emission(tag_idx, 0, feature_cache)
            else:
                for curr_tag_idx in range(num_labels):
                    marginals[word_idx][curr_tag_idx] = -np.inf
                    for prev_tag_idx in range(num_labels):
                        curr_tag = self.tag_indexer.get_object(curr_tag_idx)
                        prev_tag = self.tag_indexer.get_object(prev_tag_idx)
                        if isO(prev_tag) and isI(curr_tag):
                            continue
                        if isI(curr_tag) and (get_tag_label(curr_tag) != get_tag_label(prev_tag)):
                            continue
                        curr_p = fbss.score_emission(curr_tag_idx, word_idx, feature_cache) + \
                                 marginals[word_idx - 1][prev_tag_idx]
                        if curr_p > marginals[word_idx][curr_tag_idx]:
                            marginals[word_idx][curr_tag_idx] = curr_p
                            history[word_idx][curr_tag_idx] = prev_tag_idx

        best_seq_idx = marginals.argmax(axis=1)[sent_len-1]
        final_labels = []
        for i in range(sent_len-1, 0, -1):
            final_labels.append(self.tag_indexer.get_object(best_seq_idx))
            best_seq_idx = int(history[i][best_seq_idx])
        final_labels.reverse()
        chunks = chunks_from_bio_tag_seq(final_labels)
        return LabeledSentence(sentence_tokens, chunks)