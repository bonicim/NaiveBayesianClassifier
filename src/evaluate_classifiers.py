#!/usr/bin/python

from src.TweetData import TweetData
from src.TweetClassifier import TweetClassifier
from src.TweetProbabilities import TweetProbabilities
from os import path
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB

scikit_naive_bayes_classifiers = {
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.01),
    #   "Complement Naive Bayes": ComplementNB(alpha=0.01),
    #   "Bernoulli Naive Bayes": BernoulliNB(alpha=0.01),
}


def main():
    train_data = get_training_data()
    test_data = get_testing_data()

    evaluate_scikit_naive_bayes_classifiers(train_data, test_data)
    evaluate_in_house_naive_bayes_classifier(train_data, test_data)


def evaluate_scikit_naive_bayes_classifiers(train_data, test_data):
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train = vectorizer.fit_transform(train_data.generate_tweets())
    Y_train = train_data.generate_authors()
    X_test = vectorizer.transform(test_data.generate_tweets())
    truths = test_data.generate_authors()

    for name, clf in scikit_naive_bayes_classifiers.items():
        t0 = time()
        clf.fit(X_train, Y_train)
        train_time = time() - t0

        t0 = time()
        predictions = clf.predict(X_test)
        pred_time = time() - t0

        print_reports(truths, predictions, name, train_time, pred_time)


def evaluate_in_house_naive_bayes_classifier(train_data, test_data):
    classifier = TweetClassifier(TweetProbabilities(train_data))
    t0 = time()
    classifier.train()
    train_time = time() - t0

    t0 = time()
    predictions = [
        result[0][0] for result in classifier.classify_collection_tweets(test_data)
    ]
    pred_time = time() - t0

    print_reports(
        test_data.generate_authors(),
        predictions,
        "In-House Naive Bayes Classifier",
        train_time,
        pred_time,
    )


def get_training_data():
    return TweetData(
        path.abspath(
            path.join(path.dirname(__file__), "..", "data", "training_data.csv")
        )
    )


def get_testing_data():
    return TweetData(
        path.abspath(
            path.join(path.dirname(__file__), "..", "data", "testing_data.csv")
        )
    )


def print_reports(truths, predictions, msg, train_time, pred_time):
    print(f"\n{msg}\n")
    print(classification_report(truths, predictions))
    print(f"Confusion matrix \n {confusion_matrix(truths, predictions)}\n")
    print(f"Training time: {train_time:.4f}s")
    print(f"Classifying time: {pred_time:.4f}s\n")


if __name__ == "__main__":
    main()
