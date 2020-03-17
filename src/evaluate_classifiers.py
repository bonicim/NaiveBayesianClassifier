#!/usr/bin/python

from src.TweetData import TweetData
from src.Tweet import Tweet
from src.TweetProbabilities import TweetProbabilities
from os import path
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
        clf.fit(X_train, Y_train)
        predictions = clf.predict(X_test)
        print_reports(truths, predictions, name)


def evaluate_in_house_naive_bayes_classifier(train_data, test_data):
    classifier = Tweet(TweetProbabilities(train_data))
    classifier.train()
    predictions = [
        result[0][0] for result in classifier.classify_collection_tweets(test_data)
    ]

    print_reports(
        test_data.generate_authors(), predictions, "In-House Naive Bayes Classifier"
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


def print_reports(truths, predictions, msg):
    print(f"\n{msg}\n")
    print(classification_report(truths, predictions))
    print(f"Confusion matrix \n {confusion_matrix(truths, predictions)}\n")


if __name__ == "__main__":
    main()
