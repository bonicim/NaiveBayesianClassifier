from nltk.corpus import stopwords
from csv import reader as csvreader
from collections import Counter
from typing import Tuple, List
import string


class TweetData:
    def __init__(self, filename, stop_words=None):
        self._filename = filename
        self._tweets = self._read(filename)
        if stop_words is not None:
            self._stop_words = stop_words
        else:
            self._stop_words = set(stopwords.words("english"))

    def process(self, n_gram_size=None) -> List[Tuple]:
        if n_gram_size is None:
            n_gram_size = 1
        tweets = self._tokenize(self._tweets, n_gram_size)
        tweets = self._remove_stopwords(tweets)
        tweets = self._make_bag_of_words(tweets)

        return tweets

    def _read(self, filename):
        with open(filename) as csvfile:
            reader = csvreader(csvfile, delimiter=",", quotechar='"')
            next(reader)
            tweets = [(row[1], row[2]) for row in reader]

        return tweets

    def _tokenize(self, tweets, n_gram_size):
        remove_punctuation = lambda word: word.translate(
            str.maketrans("", "", string.punctuation)
        )

        tweets = [
            (tweet[0], [remove_punctuation(word).lower() for word in tweet[1].split()])
            for tweet in tweets
        ]

        return self._tokenize_n_gram(tweets, n_gram_size)

    # TODO: write test for n_gram_size > than 1
    def _tokenize_n_gram(self, tweets, n_gram_size):
        if n_gram_size == 1:
            return tweets

        result = []
        for author, tweet in tweets:
            re_tokenized_tweet = []
            end = len(tweet) - n_gram_size - 1
            for i in range(end):
                token = ""
                for j in range(i, n_gram_size):
                    to_add = tweet[j].ljust(len(tweet[j]))
                    token += token.join(to_add)
                re_tokenized_tweet.append(token)
            result.append((author, re_tokenized_tweet))

        return result

    def _remove_stopwords(self, tweets_tokenized):
        return [
            (tweet[0], [word for word in tweet[1] if word not in self._stop_words])
            for tweet in tweets_tokenized
        ]

    def _make_bag_of_words(self, tweets):
        return [(author, Counter(tokens)) for author, tokens in tweets]

    def generate_tweets(self) -> List[str]:
        return [tweet for _, tweet in self._tweets]

    def generate_authors(self) -> List[str]:
        return [author for author, _ in self._tweets]

    def generate_author_to_tweet_data(self) -> List[Tuple]:
        return self._tweets
