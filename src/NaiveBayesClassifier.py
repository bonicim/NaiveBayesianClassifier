import csv
import string
import re
import math

DONALD = "donald"
HILLARY = "hillary"
FILE = 'tweets.csv'
DELIMITER = ','
QUOTECHAR = '"'
SPLITTER = "\W+"
MODE = 'rb'

# A monolith program that that classifies a list of tweets by Hillary or Donald and creates a Naive Bayesian Classifier


def clean_string(tweet):
    # TWEET is a string
    # Removes any and all punctuation in TWEET; returns cleansed TWEET
    punctuation_set = set(string.punctuation)
    cleaned_tweet = ''.join(filter(lambda x: x not in punctuation_set, tweet))
    # TODO: stop words
    return cleaned_tweet


def tokenize_text(tweet):
    # TWEET is a string
    # Parses TWEET by space and converts to lowercase; returns a list of all tokens in TWEET
    tweet = clean_string(tweet)
    tweet = tweet.lower()
    list_words_in_tweet = re.split(SPLITTER, tweet)
    return list_words_in_tweet


def make_bag(tweet_list):
    # TWEET_LIST is a list of strings
    # Calculates the total frequency of each unique word in TWEET_LIST
    # Returns the results in a dictionary; key = unique word, value = frequency
    bag = {}
    for word in tweet_list:
        bag[word] = bag.get(word, 0.0) + 1.0
    return bag


def read(document):
    # DOCUMENT is a csv file
    # Cleanses each tweet entry in DOCUMENT and returns a list of cleansed tweets
    list_tweets = []

    with open(document, MODE) as csvfile:
        reader = csv.reader(csvfile, delimiter=DELIMITER, quotechar=QUOTECHAR)
        next(reader, None)
        for tweet in reader:
            # cleanse the tweet into some tuple
            tweet_tuple = (tweet[1], tweet[2])
            list_tweets.append(tweet_tuple)

    return list_tweets


def cleanup(list_tweets, category_list):
    # LIST_TWEETS is a list of tweets by both candidates
    # CATEGORY_LIST is a list of categories to classify the tweets
    # this will set and initialize all the required data structures used by train()
    # return a list of all required items

    # setup data structures to hold info
    word_tally = {}

    category_priors = {}
    for category in category_list:
        category_priors[category] = 0.0

    category_word_tally = {}
    for category in category_list:
        print category
        category_word_tally[category] = {}

    for tweet in list_tweets:
        category = tweet[0]
        cleaned_tweet = clean_string(tweet)
        list_words_in_tweet = tokenize_text(cleaned_tweet)
        bag = make_bag(list_words_in_tweet)

        print category
        category_priors[category] = category_priors.get(category) + 1

        for word, count in bag.items():
            if word not in word_tally:
                word_tally.get(word, 0.0)

            cat_word_tally = category_word_tally[category]
            if word not in cat_word_tally:
                cat_word_tally[word] = 0.0

            word_tally[word] = word_tally.get(word) + count
            cat_word_tally[word] = cat_word_tally.get(word) + count

    list_training_data = [word_tally, category_priors, category_word_tally]
    return list_training_data

# TODO: implement this
def train(document):
    # returns a numpy
    # DOCUMENT is a csv file
    list_tweets = read(document)
    cleanup(list_tweets)
    return 1


def predict():
    return 1


def evaluation():
    return


def main():
    """
    # glass box tests
    tweet = "This This is a tweet; with some punctuation, which should all be removed."
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


    # glass box test: read()
    list_all_tweets = read(FILE)
    """
    for author, tweet in list_all_tweets:
        print author
        print tweet
        print "\n"
    """

    # glass box test cleanup()
    list_category = [DONALD, HILLARY]
    list_data = cleanup(list_all_tweets, list_category)

    print "This is the following data structures from cleanup()"
    word_tally = list_data[0] # a dictionary of word to count; key:value
    category_priors = list_data[1] # a dictionary of category to count; key:value
    category_word_tally = list_data[2] # a dictionary of category to dictionary (word:count)

    print "The following is the list and frequencies of all words in document: "
    for word, count in word_tally.items():
        print "The word is: ", word, "; the word count is: ", count

    print "The following is the category list and frequency of all tweets for a category: "
    for category, count in category_priors.items():
        print "The category is: ", category, "; the tweet count is: ", count

    print "The following is the category word tally for each category: "
    for category, cat_dict in category_word_tally.items():
        print "The category is: ", category
        for word, count in cat_dict.items():
            print "The word is: ", word, "; the word count is: ", count

    return True


if __name__ == '__main__':
    main()

