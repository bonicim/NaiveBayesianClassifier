import math


class Features:
    def __init__(self, data_processor):
        self._data_processor = data_processor
        self._tweets = None

    def extract_features(self):
        self._tweets = self._data_processor.process()

        prior_prob = self._build_prior_prob()
        vocab = self._build_vocab()
        cond_prob = self._build_cond_prob()
        return (prior_prob, vocab, cond_prob)

    def _build_prior_prob(self):
        total_docs = len(self._tweets)
        prior_prob = dict()

        for author, tokens in self._tweets:
            count = prior_prob.setdefault(author, 0)
            prior_prob[author] = count + 1

        for author, count in prior_prob.items():
            prior_prob[author] = math.log(count / total_docs)

        return prior_prob

    def _build_vocab(self):
        vocab = dict()

        for _, tokens in self._tweets:
            for token, freq in tokens.items():
                count = vocab.get(token, 0)
                vocab[token] = count + freq

        return vocab

    def _build_cond_prob(self):
        # for each author build up their own vocab
        cond_prob = dict()

        # get raw count of each author's word in vocab
        for author, tweet in self._tweets:
            author_vocab = cond_prob.setdefault(author, {})
            for word, freq in tweet.items():
                author_vocab[word] = author_vocab.setdefault(word, 0) + freq

        # compute prob of each author word in vocab
        for author, tweet in cond_prob.items():
            total_words = sum(tweet.values())
            for word, freq in tweet.items():
                tweet[word] = freq / total_words

        return cond_prob
