import math


class Features:
    def __init__(self, data_processor):
        self._data_processor = data_processor
        self._documents = None

    def extract_features(self):
        self._documents = self._data_processor.process()

        prior_prob = self._build_prior_prob()
        vocab = self._build_vocab()
        cond_prob = self._build_cond_prob()
        return (prior_prob, vocab, cond_prob)

    def _build_prior_prob(self):
        total_docs = len(self._documents)
        prior_prob = dict()

        for author, tokens in self._documents:
            count = prior_prob.setdefault(author, 0)
            prior_prob[author] = count + 1

        for author, count in prior_prob.items():
            prior_prob[author] = math.log(count / total_docs)

        return prior_prob

    def _build_vocab(self):
        vocab = dict()

        for _, tokens in self._documents:
            for token, freq in tokens.items():
                count = vocab.get(token, 0)
                vocab[token] = count + freq

        return vocab

    def _build_cond_prob(self):
        # for each author build up their own vocab
        cond_prob = dict()

        for author, tokens in self._documents:
            author_vocab = cond_prob.setdefault(author, {})
            for token, freq in tokens.items():
                count = author_vocab.setdefault(token, 0)
                author_vocab[token] = count + freq

        for author, tokens in cond_prob.items():
            total_words = sum(tokens.values())
            for token, freq in tokens.items():
                tokens[token] = freq / total_words

        return cond_prob
