import csv
import string
import re
import numpy as np
from sklearn.naive_bayes import GaussianNB
import math

FILE = 'tweets.csv'
DELIMITER = ','
QUOTECHAR = '"'
SPLITTER = "\W+"
MODE = 'rb'

# A Naive Bayeasian Classfier that classifies a list of tweets based on the
# given categories in the list


def clean_tweet(tweet):
    """
    Removes any and all punctuation in TWEET; returns cleansed TWEET
    :param tweet: a string
    :return: a string that has been cleansed
    """

    punctuation_set = set(string.punctuation)
    cleaned_tweet = ''.join(filter(lambda x: x not in punctuation_set, tweet))
    # TODO: stop words, spaces and such
    return cleaned_tweet


def tokenize_tweet(tweet):
    """
    Parses TWEET by space and converts to lowercase
    :param tweet: a string
    :return: a list of all tokens in TWEET
    """

    tweet = clean_tweet(tweet)
    tweet = tweet.lower()
    tweet_list = re.split(SPLITTER, tweet)
    return tweet_list


def make_tweet_bag(tweet_list):
    """
    Calculates the total frequency of each unique word in TWEET_LIST
    :param tweet_list: a list of strings
    :return: a dictionary of key = unique word, value = frequency
    """

    tweet_bag = {}
    for word in tweet_list:
        tweet_bag[word] = tweet_bag.get(word, 0.0) + 1.0
    return tweet_bag


def read_data(document):
    """
    Creates a list of tweet tuples from DOCUMENT
    :param document: csv file of tweets
    :return: list of tweet tuples consisting of the author and tweet

        example:

        [ ("donald", "Make America Great Again."),
          ("hillary", "I'm with Her.") ]
    """

    with open(document, MODE) as csvfile:
        reader = csv.reader(csvfile, delimiter=DELIMITER, quotechar=QUOTECHAR)
        next(reader, None)
        list_tweets = map(lambda tweet: (tweet[1], tweet[2]), reader)
    return list_tweets


def get_target_values_list(list_tweets):
    """
    Creates a vector of the author for each tweet in LIST_TWEETS;
    :param list_tweets: a list of tweets in the form of a tuple

        example:

        [ (donald : "Build a wall."),
          (hillary : "Break the ceiling.") ]

    :return: a list of two objects: a numpy array of the vector of tweet
    authors and a dictionary containing the index to author mapping

        example:

        [ [0, 1, 1, 0], {0 : "donald", 1 : "hillary"} ]
    """
    index = 0
    dict_category_key = {}
    for tweet in list_tweets:
        category = tweet[0]
        if category not in dict_category_key:
            dict_category_key[category] = index
            index += 1

    list_targets = []
    for tweet in list_tweets:
        category = tweet[0]
        target = dict_category_key.get(category)
        list_targets.append(target)
    list_targets = np.array(list_targets)

    return [list_targets, dict_category_key]


def get_corpus_list(list_tweets):
    """
    Creates a list of tuples that contain every unique word and its
    associated frequency for all tweets in LIST_TWEETS
    :param list_tweets: a list of tweets in the form of a tuple

        example:

        [ (donald : "Build a wall."),
          (hillary : "Break the ceiling.") ]

    :return: a list of unique word tuples

        example:

        [ ("dog", 5), ("cat", 2) ]
    """
    dict_corpus = {}
    for tweet in list_tweets:
        # turn tweet into a bag of words
        tweet_text = tweet[1]
        cleaned_tweet = clean_tweet(tweet_text)
        list_words_in_tweet = tokenize_tweet(cleaned_tweet)
        bag = make_tweet_bag(list_words_in_tweet)

        # update the total tally for each word
        for word, count in bag.items():
            if word not in dict_corpus:
                dict_corpus[word] = 0.0
            dict_corpus[word] = dict_corpus.get(word) + count

    list_corpus = []
    for key, value in dict_corpus.items():
        temp = [key, value]
        list_corpus.append(temp)

    return list_corpus


def get_tweet_list_bag(list_tweets):
    """
    Converts each tweet into a bag; creates a list of all tweet bags and
    their associated author
    :param list_tweets: a list of tweets in the form of a tuple

        example:

        [ (donald : "Build a wall."),
          (hillary : "Break the ceiling.") ]

    :return: a list of tweet bags and associated authors

        example:

        [ ("donald", {"wall" : 5, "rich" : 2}),
          ("hillary", {"glass" : 5, "emails" : 2, "poor" : 8}) ]
    """
    list_tweet_bag = []

    for tweet in list_tweets:
        tweet_author = tweet[0]
        tweet_text = tweet[1]
        cleaned_tweet = clean_tweet(tweet_text)
        list_words_in_tweet = tokenize_tweet(cleaned_tweet)
        bag = make_tweet_bag(list_words_in_tweet)
        tweet_tuple = (tweet_author, bag)
        list_tweet_bag.append(tweet_tuple)

    return list_tweet_bag


def get_tweet_list_word_count(corpus, tweet_list_bag):
    """
    Creates a list of word frequency for every tweet in TWEET_LIST_BAG
    based upon the master word list in CORPUS.
    :param corpus: a list of unique word tuples

        example:

        [ ("dog", 5), ("cat", 2) ]

    :param tweet_list_bag: a list of tweet bags and associated authors

        example:

        [ ("donald", {"wall" : 5, "rich" : 2}),
          ("hillary", {"glass" : 5, "emails" : 2, "poor" : 8}) ]

    :return: a numpy array object of the list of word frequencies of all tweets

        example:

        [ [3, 0, 5, 6, 7]
          [13, 60, 53, 2, 0]
          [39, 0, 5, 0, 0] ]
    """
    list_master = []

    for tweet in tweet_list_bag:
        bag_dict = tweet[1]
        list_tweet_val = []
        for word_tuple in corpus:
            word = word_tuple[0]
            val = 0.0
            if word in bag_dict:
                val = bag_dict[word]
            list_tweet_val.append(val)
        list_master.append(list_tweet_val)

    numpy_arr = np.array(list_master)
    return numpy_arr


def get_scikit_fit_args(list_tweets):
    """
    Creates the two arguments required for scikit's GaussianNB object's
    "fit(X, y" method.
    :param list_tweets: a list of tweets in the form of a tuple

        example:

        [ (donald : "Build a wall."),
          (hillary : "Break the ceiling.") ]

    :return: a list of the two arguments required for GaussianNB.fit(X, y).

        example:

        [ [ [3, 0, 5, 6, 7]
          [13, 60, 53, 2, 0]
          [39, 0, 5, 0, 0] ],

           [0, 1, 1, 0] ]

    """
    corpus = get_corpus_list(list_tweets)
    tweet_list_bag = get_tweet_list_bag(list_tweets)
    np_array = get_tweet_list_word_count(corpus, tweet_list_bag)
    np_x_arg = np_array

    list_target_val = get_target_values_list(list_tweets)
    np_y_arg = list_target_val[0]

    args = [np_x_arg, np_y_arg]
    return args


def get_corpus_prob_dict(corpus_list):
    # TODO: Implement
    """
    Calculates probabilities for every word in CORPUS_LIST
    :param corpus_list: a list of unique word tuples

        example:

        [ ("dog", 5), ("cat", 2) ]

    :return: a dictionary comprised of word to probability pairs
    """
    dict_corpus_prob = {}
    return dict_corpus_prob


def get_author_prob_dict(list_target_values):
    # TODO: Implement
    """
    Calculates probabilities for every author in LIST_TARGET_VALUES
    :param list_target_values: a list of two objects: a numpy array of the
    vector of tweet authors and a dictionary containing the index to author
    mapping

        example:

        [ [0, 1, 1, 0], {0 : "donald", 1 : "hillary"} ]

    :return: a dictionary comprised of author to probability pairs
    """

    dict_author_prob = {}
    return dict_author_prob


def get_cond_prob_dict(corpus_list, tweet_np_matrix, target_val_list):
    # TODO: Implement
    """
    Calculates conditional probabilities for all categories in TARGET_VAL_LIST
    :param corpus_list: a list of unique word tuples

        example:

        [ ("dog", 5), ("cat", 2) ]

    :param tweet_np_matrix: a numpy array object of the list of
    word frequencies of all tweets

        example:

        [ [3, 0, 5, 6, 7]
          [13, 60, 53, 2, 0]
          [39, 0, 5, 0, 0] ]

    :param target_val_list: a list of two objects: a numpy array
    of the vector of tweet authors and a dictionary containing the index to
    author mapping

        example:

        [ [0, 1, 1, 0], {0 : "donald", 1 : "hillary"} ]
    :return: a dictionary comprised of author to dictionary pairs. The
    dictionary value is comprised of word to probability pairs.

        example:

        { "donald" :  { "wall" : .23, "great" : .009},
          "hillary" : { "predator" : .32, "democrat" : .119} }
    """
    dict_author_cond_prob = {}
    return dict_author_cond_prob


def cleanup(list_tweets):
    """
    Cleans all tweets and calculates three data sets needed for Naive
    Bayesian analysis

    :param list_tweets: list of tweets from all candidates
    :return: list of 3 sets of data;

    The first, LIST_CORPUS, is a list that holds frequencies of all
    words in LIST_TWEETS.

    The second, LIST_TARGET_VALUES, is a list of two objects:
    a numpy array of the vector of tweet authors and a dictionary
    containing the index to author mapping.

        example:

        [ [0, 1, 1, 0], {0 : "donald", 1 : "hillary"} ]

    The third, TWEET_NP_MATRIX, is a numpy array object of the list of word
    frequencies of all tweets

        example:

        [ [3, 0, 5, 6, 7]
          [13, 60, 53, 2, 0]
          [39, 0, 5, 0, 0] ]
    """

    list_corpus = get_corpus_list(list_tweets)
    list_target_values = get_target_values_list(list_tweets)
    tweet_list_bag = get_tweet_list_bag(list_tweets)
    tweet_np_matrix = get_tweet_list_word_count(list_corpus, tweet_list_bag)

    return [list_corpus, list_target_values, tweet_np_matrix]


def train(document):
    """
    :param document: csv file of tweets
    :return: a list consisting of three data:

    The first is a dictionary comprised of word to probability pairs.

    The second is a dictionary comprised of author to probability pairs.

    The third is a dictionary comprised of author to dictionary pairs. The
    dictionary value is comprised of word to probability pairs.

        example:

        { "donald" :  { "wall" : .23, "great" : .009},
          "hillary" : { "predator" : .32, "democrat" : .119} }
    """

    list_tweets = read_data(document)
    list_data = cleanup(list_tweets)

    list_corpus = list_data[0]
    list_target_values = list_data[1]
    tweet_np_matrix = list_data[2]

    dict_corpus_prob = get_corpus_prob_dict(list_corpus)
    dict_author_prob = get_author_prob_dict(list_target_values)
    dict_cond_prob = get_cond_prob_dict(list_corpus, tweet_np_matrix,
                                        list_target_values)

    data_set = [dict_corpus_prob, dict_author_prob, dict_cond_prob]
    return data_set


def predict(dataset, tweet):
    # TODO: implement and test
    """
    Calculates and returns the Naive Bayesian score for TWEET based on DATASET
    :param dataset: a list consisting of three data:

    The first is a dictionary comprised of word to probability pairs.

    The second is a dictionary comprised of author to probability pairs.

    The third is a dictionary comprised of author to dictionary pairs. The
    dictionary value is comprised of word to probability pairs.

        example:

        { "donald" :  { "wall" : .23, "great" : .009},
          "hillary" : { "predator" : .32, "democrat" : .119} }

    :param tweet: a string
    :return: list of classes that TWEET might belong to with decreasing
    probability
    """
    list_prob_of_tweet = []
    dict_corpus_prob = dataset[0]
    dict_author_prob = dataset[1]
    dict_cond_prob = dataset[2]

    # create score accumulators

    # DO SOME MATH and accumulate scores

    return list_prob_of_tweet


def evaluation(data_set):
    # TODO: implement
    """
    Computes the overall accuracy and the per class accuracy of classifier
    Precision - Out of all the answers by the classifier, which ones are correct?
    Recall - Out of all the actual correct answers, which ones are returned by the classifier?
    F1-Score - 2 * precision * reecall / (precision + recall)

    :param data_set:

    :return:
    """
    return data_set


def tests():
    """
    Black and glass box testing for all methods
    :return: True
    """
    list_tweets = read_data(FILE)

    # black box test: cleanup()
    list_prob_args = cleanup(list_tweets)

    list_corpus = list_prob_args[0]
    list_target_values = list_prob_args[1]
    tweet_np_matrix = list_prob_args[2]

    # glass box test: get_corpus_prob_dict()
    dict_corpus_prob = get_corpus_prob_dict(list_corpus)
    print dict_corpus_prob

    # glass box test: get author_prob_dict()
    dict_author_prob = get_author_prob_dict(list_target_values)
    print dict_author_prob

    # glass box test: get_cond_prob_dict()
    dict_cond_prob = get_cond_prob_dict(list_corpus, tweet_np_matrix,
                                        list_target_values)
    print dict_cond_prob

    # black box test: train()
    data_set = train(FILE)
    for data in data_set:
        print data

    """
     # glass box test: get_scikit_fit_args()
    list_scikit_fit_args = get_scikit_fit_args(list_tweets)
    print list_scikit_fit_args[1]
    print list_scikit_fit_args[0]
    """

    """
    # glass box test: get_tweet_list_word_count()
    corpus = get_corpus(list_tweets)
    tweet_list_bag = get_tweet_list_bag(list_tweets)
    np_array = get_tweet_list_word_count(corpus, tweet_list_bag)
    print np_array
    """

    """
    # glass box test: get_tweet_list_bag()
    list_tweet_bag = get_tweet_list_bag(list_tweets)

    for tweet in list_tweet_bag:
        print tweet
        break

    for tweet in list_tweets:
        print tweet
        break
    """

    """
    # glass box test: get_corpus()
    list_corpus = get_corpus(list_tweets)
    for tweet in list_tweets:
        print tweet
        break
    print list_corpus
    """

    """
    # glass box test: get_target_values()
    list_targ_val = get_target_values(list_tweets)
    print list_targ_val[0]
    print list_targ_val[1]
    """

    """
    # glass box tests
    tweet = "This is a tweet; this tweet, which should not have punc!"
    cleaned_tweet = clean_string(tweet)
    print "Testing clean_string()"
    print "This is the original tweet: ", tweet
    print "This is the cleaned tweet: ", cleaned_tweet

    list_words_in_tweet = tokenize_text(cleaned_tweet)
    print "Testing tokenize_text()"
    for word in list_words_in_tweet:
        print "token: ", word

    bag = make_bag(list_words_in_tweet)
    print "Testing make_bag()"
    for word, freq in bag.items():
        print "word: ", word
        print "freq: ", freq
    """

    """
    # glass box test: read()
    list_all_tweets = read(FILE)

    for author, tweet in list_all_tweets:
        print author
        print tweet
        print "\n"
    """

    """
    # glass box test cleanup()
    list_data = cleanup(list_all_tweets)
    word_tally = list_data[0]  # a dictionary of word to count
    category_tweet_count = list_data[1]  # a dictionary of category to count
    category_word_tally = list_data[2]  # a dictionary of category to new dict

    print "Success", "\n"
    print "This is the following data structures from cleanup()", "\n"

    print "The following is list and frequencies of all words in document: "
    stop = 0
    for word, count in word_tally.items():
        print "The word is: ", word, "; the word count is: ", count
        stop += 1
        if stop > 10:
            stop = 0
            break

    print "\n", "The following is the category list and frequency of all " \
          "tweets " \
          "per category: "
    for category, count in category_tweet_count:
        print "The category is: ", category, "; the tweet count is: ", count

    print "\n", "The following is the category word tally for each category: "
    for category, cat_dict in category_word_tally.items():
        print "\n", "The category is: ", category, "\n"
        for word, count in cat_dict.items():
            print "The word is: ", word, "; the word count is: ", count
            stop += 1
            if stop > 10:
                stop = 0
                break
    """

    """
    # glassbox test of train()
    result = train(FILE)
    """
    return True


def main():
    return tests()

if __name__ == '__main__':
    main()

