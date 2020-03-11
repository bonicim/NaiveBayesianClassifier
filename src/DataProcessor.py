from csv import reader as csvreader


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
        if self._tweets is None:
            self.read()

        tweets_tokenized = [(tweet[0], tweet[1].split()) for tweet in self._tweets]

        return tweets_tokenized
