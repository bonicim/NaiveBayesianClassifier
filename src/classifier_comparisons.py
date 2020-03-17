#!/usr/bin/python

from src.Tweet import Tweet
from src.TweetProbabilities import TweetProbabilities
from src.TweetData import TweetData

import numpy as np
import csv
from os import path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


TEST_TARGET = ["HillaryClinton", "realDonaldTrump"]


def main():
    d = TweetData(
        path.abspath(
            path.join(path.dirname(__file__), "..", "data", "training_data.csv")
        )
    )

    f = TweetProbabilities(
        TweetData(
            path.abspath(path.join(path.dirname(__file__), "..", "data", "tweets.csv"))
        )
    )
    prior_prob, vocab, cond_prob = f.extract_features()

    categories = {}
    x = -1
    for author in prior_prob.keys():
        x += 1
        categories[author] = x

    # creates a list of lists where the first number maps to an author and the following numbers map to the count of every word in the vocabulary that was seen in the current tweet
    # result = []
    # for author, tweet in tweets:
    #     tweet_val = []
    #     tweet_val.append(categories.get(author))

    #     # this is expensive might be a better way
    #     for word, count in vocab.items():
    #         val = 0.0
    #         if word in tweet:
    #             val += count
    #         tweet_val.append(val)

    #     result.append(tweet_val)

    # numpy_arr = np.array(result)
    # numpy_shape = np.shape(numpy_arr)
    # training_data = numpy_arr[:, 1 : numpy_shape[1]]
    # training_targets = numpy_arr[:, 0]

    # # build a SciKit classifier
    # text_classfier = Pipeline(
    #     [
    #         ("vect", CountVectorizer()),
    #         ("tfidf", TfidfTransformer()),
    #         ("clf", MultinomialNB()),
    #     ]
    # )

    # print(training_data)
    # print(training_targets)
    # text_classfier = text_classfier.fit(training_data, training_targets)
    # prediction = text_classfier.predict(d.generate_tweet_data())

    # score = np.mean(prediction == training_targets)

    # # printing reports
    # print(f"Score: {score}")
    # print(
    #     metrics.classification_report(
    #         TEST_TARGET, prediction, target_names=TEST_TARGET
    #     )
    # )
    # print(f"Confusion matrix\n")
    # print(metrics.confusion_matrix(TEST_TARGET, prediction))


if __name__ == "__main__":
    # execute only if run as the entry point into the program
    main()
