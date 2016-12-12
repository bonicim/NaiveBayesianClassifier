import NaiveBayesClassifier as nbc
import numpy as np
import csv
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

TRAINING_DATA_FILE = 'tweets.csv'
MODE = 'rb'
DELIMITER = ','
QUOTECHAR = '"'
SPLITTER = "\W+"
TEST_DATA = 'test_data.csv'
TEST_TARGET = ['HillaryClinton', 'realDonaldTrump']
TEST_SMALL_DATA = ["The question in this election: Who can put the plans "
                   "into action that will make your life better?",
                   "It wasn't Matt Lauer that hurt Hillary last night. It was her very dumb answer about emails &amp; the veteran who said she should be in jail."]


def main():
    # get data and targets
    with open(TRAINING_DATA_FILE, MODE) as csvfile:
        reader = csv.reader(csvfile, delimiter=DELIMITER, quotechar=QUOTECHAR)
        next(reader, None)
        list_tweets = map(lambda tweet: (tweet[1], tweet[2]), reader)
        test_data_list = map(lambda tweet: tweet[2], reader)

    args = nbc.get_scikit_fit_args(list_tweets)

    data_x = args[0]
    target_y = args[1]

    # build classifier
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                         ])
    text_clf = text_clf.fit(data_x, target_y)

    # test the classfier against some testing data

    predicted_target = text_clf.predict(test_data_list)
    score = np.mean(predicted_target == target_y)

    # printing reports
    print "The score is: ", score
    print(metrics.classification_report(TEST_TARGET, predicted_target,
                                        target_names=TEST_TARGET))
    print "\n", "Confusion matrix: ", "\n"
    print metrics.confusion_matrix(TEST_TARGET, predicted_target)


if __name__ == '__main__':
    main()


