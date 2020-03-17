from src.TweetData import TweetData
from typing import Tuple
import math


class TweetFeatures:
    def __init__(self, tweet_data):
        self._tweet_data = tweet_data
        self._tweets = None

    def extract_features(self) -> Tuple:
        self._tweets = self._tweet_data.process()
        # prior probability of every author
        prior_prob = self._build_prior_prob()
        # raw count of every word in all tweets
        vocab = self._build_vocab()
        # word probabilities per author
        cond_prob = self._build_feature_probabilities()
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

    def _build_feature_probabilities(self):
        cond_prob = dict()

        # get raw count of each author's word in vocab
        for author, tweet in self._tweets:
            author_vocab = cond_prob.setdefault(author, {})
            for word, freq in tweet.items():
                author_vocab[word] = author_vocab.setdefault(word, 0) + freq

        # compute prob of each author word in vocab
        for author, author_vocab in cond_prob.items():
            total_words = sum(author_vocab.values())
            for word, freq in author_vocab.items():
                author_vocab[word] = freq / total_words

        return cond_prob
