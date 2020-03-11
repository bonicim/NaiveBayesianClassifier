from csv import reader as csvreader
import string


class DataProcessor:
    def __init__(self, filename):
        self._filename = filename
        self._tweets = None

    def read(self):
        with open(self._filename) as csvfile:
            reader = csvreader(csvfile, delimiter=",", quotechar='"')
            next(reader)
            tweets = [(row[1], row[2]) for row in reader]
        self._tweets = tweets
        return tweets

    def tokenize(self):
        remove_punctuation = lambda word: word.translate(
            str.maketrans("", "", string.punctuation)
        )
        if self._tweets is None:
            self.read()

        tweets_tokenized = [
            (tweet[0], [remove_punctuation(word).lower() for word in tweet[1].split()])
            for tweet in self._tweets
        ]

        return tweets_tokenized
