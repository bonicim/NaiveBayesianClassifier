from nltk.corpus import stopwords
from csv import reader as csvreader
import string


class DataProcessor:
    def __init__(self, filename):
        self._filename = filename

    def process(self):
        tweets = self._read()
        tweets = self._tokenize(tweets)
        tweets = self._remove_stopwords(tweets)

        return tweets

    def _read(self):
        with open(self._filename) as csvfile:
            reader = csvreader(csvfile, delimiter=",", quotechar='"')
            next(reader)
            tweets = [(row[1], row[2]) for row in reader]

        return tweets

    def _tokenize(self, tweets):
        remove_punctuation = lambda word: word.translate(
            str.maketrans("", "", string.punctuation)
        )

        return [
            (tweet[0], [remove_punctuation(word).lower() for word in tweet[1].split()])
            for tweet in tweets
        ]

    def _remove_stopwords(self, tweets_tokenized):
        stop_words = set(stopwords.words("english"))
        return [
            (tweet[0], [word for word in tweet[1] if word not in stop_words])
            for tweet in tweets_tokenized
        ]
