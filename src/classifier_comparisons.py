#!/usr/bin/python

from src.TweetData import TweetData
from os import path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB


def main():
    d = TweetData(
        path.abspath(
            path.join(path.dirname(__file__), "..", "data", "training_data.csv")
        )
    )
    d.process()

    vectorizer = TfidfVectorizer(stop_words="english")
    X_train = vectorizer.fit_transform(d.generate_tweets())
    Y_train = d.generate_tweet_authors()

    test_data = TweetData(
        path.abspath(
            path.join(path.dirname(__file__), "..", "data", "testing_data.csv")
        )
    )
    test_data.process()

    X_test = vectorizer.transform(test_data.generate_tweets())

    mnb = MultinomialNB(alpha=0.01)

    mnb.fit(X_train, Y_train)
    predictions = mnb.predict(X_test)

    print(predictions)


if __name__ == "__main__":
    main()
