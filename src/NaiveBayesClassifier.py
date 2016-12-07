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


def read(document):
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


def get_target_values(list_tweets):
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


def get_corpus(list_tweets):
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
    corpus = get_corpus(list_tweets)
    tweet_list_bag = get_tweet_list_bag(list_tweets)
    np_array = get_tweet_list_word_count(corpus, tweet_list_bag)
    np_x_arg = np_array

    list_target_val = get_target_values(list_tweets)
    np_y_arg = list_target_val[0]

    args = [np_x_arg, np_y_arg]
    return args


def cleanup(list_tweets):
    """
    :param list_tweets: list of tweets from all candidates
    :return: list of 3 sets of data;

    The first, WORD_TALLY, is a dictionary that holds frequencies of all
    words in LIST_TWEETS

    The second, CATEGORY_TWEET_COUNT, is a list of tuples; the tuple has two
    members: the category and the total tweets associated with the category

        example:

        (donald, 846)
        (hillary, 987)

    The third, CATEGORY_WORD_TALLY, is a dictionary of categories and their
    associated word tally dictionaries.

        example of one key, value pair:

        donald : {great : 12, again: 123, america : 84}
        hillary : {women : 94723, love: 1731, glass: 854}
    """

    word_tally = {}
    category_tweet_count = {}
    category_word_tally = {}

    for tweet in list_tweets:
        category = tweet[0]
        if category not in category_tweet_count:
            category_tweet_count[category] = 0.0
        if category not in category_word_tally:
            category_word_tally[category] = {}

        # turn tweet into a bag of words
        cleaned_tweet = clean_tweet(tweet)
        list_words_in_tweet = tokenize_tweet(cleaned_tweet)
        bag = make_tweet_bag(list_words_in_tweet)

        # update appropriate category tweet count
        category_tweet_count[category] = category_tweet_count.get(category) + 1

        # update the total tally for each word
        # update the category's total tally for each word
        for word, count in bag.items():
            if word not in word_tally:
                word_tally[word] = 0.0

            cat_word_tally = category_word_tally[category]
            if word not in cat_word_tally:
                cat_word_tally[word] = 0.0

            word_tally[word] = word_tally.get(word) + count
            cat_word_tally[word] = cat_word_tally.get(word) + count

    category_tweet_count = map(lambda x: (x, category_tweet_count.get(x)),
                          category_tweet_count)

    list_training_data = [word_tally, category_tweet_count, category_word_tally]
    return list_training_data


def train(document):
    """
    TODO: implement this
    :param document: csv file of tweets
    :return:
    """

    list_tweets = read(document)
    list_data = cleanup(list_tweets)

    word_tally = list_data[0]
    category_tweet_count = list_data[1]
    category_word_tally = list_data[2]

    # calculate

    # calculate prior probabilities for every category
    tweet_count = reduce(lambda x, y: x[1] + y[1], category_tweet_count)
    list_cat_prior_prob = map(lambda x: (x[0], x[1]/tweet_count),
                              category_tweet_count)

    # calculate

    return True


def predict():
    return 1


def evaluation():
    return


def main():
    list_tweets = read(FILE)

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


if __name__ == '__main__':
    main()

