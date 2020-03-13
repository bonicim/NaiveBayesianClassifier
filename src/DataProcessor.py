from nltk.corpus import stopwords
from csv import reader as csvreader
import string
from collections import Counter


class DataProcessor:
    def __init__(self, filename):
        self._filename = filename

    def process(self, n_gram_size=None):
        if n_gram_size is None:
            n_gram_size = 1
        tweets = self._read()
        tweets = self._tokenize(tweets, n_gram_size)
        tweets = self._remove_stopwords(tweets)
        tweets = self._make_bag_of_words(tweets)

        return tweets

    def _read(self):
        with open(self._filename) as csvfile:
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

        tweets = self._tokenize_n_gram(tweets, n_gram_size)

        return tweets

    # TODO: write test for size greater than 1
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
        stop_words = set(stopwords.words("english"))
        return [
            (tweet[0], [word for word in tweet[1] if word not in stop_words])
            for tweet in tweets_tokenized
        ]

    def _make_bag_of_words(self, tweets):
        return [(author, Counter(tokens)) for author, tokens in tweets]

    def generate_tweet_data(self):
        tweets = self._read()
        return [tweet for _, tweet in tweets]
