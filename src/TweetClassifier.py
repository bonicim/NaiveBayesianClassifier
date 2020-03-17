from src.TweetData import TweetData
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from math import log, exp
from typing import List, Tuple, Type


class TweetClassifier:
    def __init__(self, TweetProbabilities):
        self._tweet_probabilities = TweetProbabilities
        self._prior_prob = None
        self._vocab = None
        self._cond_prob = None

    def train(self):
        self._prior_prob, self._vocab, self._cond_prob = (
            self._tweet_probabilities.extract_features()
        )

    def classify(self, tweet: str) -> List[Tuple]:
        # turn tweet into bag of words
        tweet_words = Counter(tweet.split())
        hypothesis_prob = {}
        total_vocab = sum(self._vocab.values())

        # calculate the prob for each category
        for author, prob in self._prior_prob.items():
            hypothesis_prob[author] = 0.0
            author_vocab = self._cond_prob[author]
            total_author_vocab = sum(author_vocab.values())

            for word, count in tweet_words.items():
                # Adding 1 to the numerator to use Laplace smoothing to account for words not present in the author's vocab
                prob_given_author = (author_vocab.get(word, 0.0) + 1) / (
                    total_author_vocab + total_vocab
                )
                author_prob = hypothesis_prob[author]
                author_prob += log(prob_given_author)
                hypothesis_prob[author] = author_prob

        return list(sorted(hypothesis_prob.items(), key=lambda t: t[1], reverse=True))

    def classify_collection_tweets(self, tweets: Type[TweetData]) -> List[Tuple]:
        return [
            self.classify(tweet) for _, tweet in tweets.generate_author_to_tweet_data()
        ]
