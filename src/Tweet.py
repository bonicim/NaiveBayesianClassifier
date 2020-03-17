from collections import Counter
from math import log, exp
from sklearn.metrics import classification_report, confusion_matrix
import pprint
from typing import List, Tuple, Type
from src.TweetData import TweetData


class Tweet:
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
        words = Counter(tweet.split())
        hypothesis_prob = {}
        total_vocab = sum(self._vocab.values())

        # calculate the prob for each category
        for author, prob in self._prior_prob.items():
            hypothesis_prob[author] = 0.0
            author_vocab = self._cond_prob[author]
            total_author_vocab = sum(author_vocab.values())

            for word, count in words.items():
                prob_given_category = (author_vocab.get(word, 0.0) + 1) / (
                    total_author_vocab + total_vocab
                )

                if prob_given_category > 0:
                    author_prob = hypothesis_prob[author]
                    author_prob += log(prob_given_category)
                    hypothesis_prob[author] = author_prob

        hypothesis_prob = sorted(
            hypothesis_prob.items(), key=lambda t: t[1], reverse=True
        )

        return [(author, exp(prob)) for author, prob in hypothesis_prob]

    def classify_collection_tweets(self, tweets: Type[TweetData]) -> List[Tuple]:
        return [
            self.classify(tweet) for _, tweet in tweets.generate_author_tweet_data()
        ]

    def evaluation(self, test_data: Type[TweetData]):
        truths = test_data.generate_tweet_author_data()

        predictions = [
            self.classify(tweet)[0][0]
            for _, tweet in test_data.generate_author_tweet_data()
        ]

        hillary_truths = sum([1 for author in truths if author == "HillaryClinton"])
        hillary_predictions = sum(
            [1 for author in predictions if author == "HillaryClinton"]
        )

        don_truths = sum([1 for author in truths if author == "realDonaldTrump"])
        don_predictions = sum(
            [1 for author in predictions if author == "realDonaldTrump"]
        )

        print(f"\nHillary predictions: {hillary_predictions}")
        print(f"Hillary truths: {hillary_truths}")
        print(f"don predictions: {don_predictions}")
        print(f"don truths: {don_truths}")

        confusion = confusion_matrix(truths, predictions)
        classification = classification_report(truths, predictions)

        return (confusion, classification)
