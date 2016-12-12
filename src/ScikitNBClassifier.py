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


def main():
    # get data and targets
    with open(TRAINING_DATA_FILE, MODE) as csvfile:
        reader = csv.reader(csvfile, delimiter=DELIMITER, quotechar=QUOTECHAR)
        next(reader, None)
        list_tweets = map(lambda tweet: (tweet[1], tweet[2]), reader)

    data = map(lambda x: x[1], list_tweets)
    target = map(lambda x: x[0], list_tweets)

    # build classifier
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                         ])
    text_clf = text_clf.fit(data, target)

    # test the classfier against some testing data
    with open(TEST_DATA, MODE) as csvfile:
        reader = csv.reader(csvfile, delimiter=DELIMITER, quotechar=QUOTECHAR)
        next(reader, None)
        test_data_list = map(lambda tweet: tweet[1], reader)

    predicted_target = text_clf.predict(test_data_list)
    score = np.mean(predicted_target == TEST_TARGET)

    # printing reports
    print "The score is: ", score
    print(metrics.classification_report(TEST_TARGET, predicted_target,
                                        target_names=TEST_TARGET))
    print "\n", "Confusion matrix: ", "\n"
    print metrics.confusion_matrix(TEST_TARGET, predicted_target)


if __name__ == '__main__':
    main()


