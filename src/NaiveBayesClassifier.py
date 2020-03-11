import csv
import string
import re
import numpy as np
import math
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from nltk.corpus import stopwords


FILE = "tweets.csv"
DELIMITER = ","
QUOTECHAR = '"'
SPLITTER = "\W+"
MODE = "r"
SMALL_PROB = 0.00000001
TEST_DONALD = (
    "Great afternoon in Little Havana with Hispanic community "
    "leaders. Thank you for your support!"
)
TEST_HILLARY = (
    "The question in this election: Who can put the plans into "
    "action that will make your life better?"
)
TEST_DATA = "test_data.csv"
TEST_TARGET = ["HillaryClinton", "realDonaldTrump"]


# A Naive Bayeasian Classfier that classifies a list of tweets based on the
# given categories in the list


def clean_tweet(tweet):
    """
    Removes any and all punctuation in TWEET; returns cleansed TWEET
    :param tweet: a string
    :return: a string that has been cleansed
    """
    cleaned_tweet = tweet.lower()
    cleaned_tweet = re.sub(r"http\S+", "", cleaned_tweet)
    cleaned_tweet = " ".join(cleaned_tweet.split())
    punctuation_set = set(string.punctuation)
    cleaned_tweet = "".join(filter(lambda x: x not in punctuation_set, cleaned_tweet))
    stop_words = set(stopwords.words("english"))
    cleaned_tweet = filter(lambda x: x not in stop_words, cleaned_tweet.split())
    cleaned_tweet = " ".join(cleaned_tweet)
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


def read_test_data(document):
    with open(document, MODE) as csvfile:
        reader = csv.reader(csvfile, delimiter=DELIMITER, quotechar=QUOTECHAR)
        next(reader, None)
        list_tweets = map(lambda tweet: (tweet[0], tweet[1]), reader)
    return list_tweets


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

        [ ("donald", {"wall" : 1, "build" : 1}),
          ("hillary", {"break" : 1, "ceiling" : 1, "the" : 1}) ]
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


def get_author_tgt_dict(list_tweets):
    """
    Creates a dictionary of index to class pairs from LIST_TWEETS.
    :param list_tweets: a list of tweets in the form of a tuple

        example:

        [ (donald : "Build a wall."),
          (hillary : "Break the ceiling.") ]

    :return: a dictionary containing all index to class pairs.

        example:

        { "donald": 0, "hillary": 1}
    """
    tgt = 0
    dict_author_tgt = {}
    for tweet in list_tweets:
        author = tweet[0]
        if author not in dict_author_tgt:
            dict_author_tgt[author] = tgt
            tgt += 1
    return dict_author_tgt


def get_tweet_list_frequencies_numpy_array(
    list_corpus, list_tweet_bag, dict_author_tgt
):
    """
    Creates a list of word frequency for every tweet in LIST_TWIST_BAG
    based upon the master word list in CORPUS.
    :param list_corpus: a list of unique word tuples

        example:

        [ ("dog", 5), ("cat", 2) ]

    :param list_tweet_bag: a list of tweet bags and associated authors

        example:

        [ ("donald", {"wall" : 5, "rich" : 2}),
          ("hillary", {"glass" : 5, "emails" : 2, "poor" : 8}) ]

    :param dict_author_tgt:

    :return: a numpy array object of the list of author target values
    and word frequencies of all tweets

        example:

        [ [0, 3, 0, 5, 6, 7]
          [1, 13, 60, 53, 2, 0]
          [0, 39, 0, 5, 0, 0] ]
    """
    list_master = []  # list of lists

    for tweet in list_tweet_bag:
        # mark the tweet with the assigned author
        list_tweet_val = []
        bag_author = tweet[0]
        if bag_author in dict_author_tgt:
            tgt = dict_author_tgt.get(bag_author)
            list_tweet_val.append(tgt)

        tweet_bag = tweet[1]
        for word_tuple in list_corpus:
            word = word_tuple[0]
            val = 0.0
            if word in tweet_bag:
                val = tweet_bag[word] + 0.0
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
    corpus_list = get_corpus_list(list_tweets)
    tweet_list_bag = get_tweet_list_bag(list_tweets)
    author_tgt_dict = get_author_tgt_dict(list_tweets)
    numpy_array = get_tweet_list_frequencies_numpy_array(
        corpus_list, tweet_list_bag, author_tgt_dict
    )
    numpy_shape = np.shape(numpy_array)
    width = numpy_shape[1]
    fit_x_arg = numpy_array[:, 1:width]
    fit_y_arg = numpy_array[:, 0]
    return [fit_x_arg, fit_y_arg]


def get_corpus_prob_dict(corpus_list):
    """
    Calculates probabilities for every word in CORPUS_LIST
    :param corpus_list: a list of unique word tuples

        example:

        [ ("dog", 5), ("cat", 2) ]

    :return: a dictionary comprised of word to probability pairs
    """
    dict_corpus_prob = {}
    temp_list = map(lambda x: x[1], corpus_list)
    total_count = sum(temp_list)

    for word_tuple in corpus_list:
        word_key = word_tuple[0]
        word_freq = word_tuple[1]
        prob_value = word_freq / total_count
        dict_corpus_prob[word_key] = prob_value

    return dict_corpus_prob


def get_author_prob_dict(list_tweets):
    """
    Calculates probabilities for every author in LIST_TARGET_VALUES

    :param list_tweets: list of tweet tuples consisting of the author and tweet

        example:

        [ ("donald", "Make America Great Again."),
          ("hillary", "I'm with Her.") ]

    :param dict_author_tgt:

        example:

        { "donald": 0, "hillary": 1}

    :return: a dictionary comprised of author to probability pairs
    """

    dict_author_prob = {}
    total_tweets = len(list_tweets)

    for tweet in list_tweets:
        cur_author = tweet[0]
        if cur_author in dict_author_prob:
            dict_author_prob[cur_author] = dict_author_prob.get(cur_author) + 1.0
        else:
            dict_author_prob[cur_author] = 1.0

    for author, count in dict_author_prob.items():
        dict_author_prob[author] = count / total_tweets
    return dict_author_prob


def get_tgt_author_dict(dict_author_tgt):
    temp_list = map(lambda x: (dict_author_tgt.get(x), x), dict_author_tgt)
    dict_tgt_author = {}
    for item in temp_list:
        dict_tgt_author[item[0]] = item[1]
    return dict_tgt_author


def get_cond_prob_dict(corpus_list, tweet_np, dict_author_tgt):
    """
    Calculates conditional probabilities for all categories in TARGET_VAL_LIST
    :param corpus_list: a list of unique word tuples

        example:

        [ ("dog", 5), ("cat", 2) ]

    :param tweet_np: a numpy array object of the list of
    word frequencies of all tweets

        example:

        [ [3, 0, 5, 6, 7]
          [13, 60, 53, 2, 0]
          [39, 0, 5, 0, 0] ]

    :param dict_author_tgt: a list of two objects: a numpy array
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
    dict_tgt_author = get_tgt_author_dict(dict_author_tgt)
    dict_author_cond_prob = {}
    numpy_shape = np.shape(tweet_np)
    width = numpy_shape[1]

    for tgt, author in dict_tgt_author.items():
        list_tgt_filtered = tweet_np[tweet_np[:, 0] == tgt]
        numpy_author_freq = np.array(list_tgt_filtered[:, 1:width])
        collapsed_numpy_author_freq = np.sum(numpy_author_freq, axis=0)
        total_word_freq = np.sum(numpy_author_freq)

        dict_word_prob = {}
        for i in range(len(corpus_list)):
            word_key = corpus_list[i][0]
            word_count = collapsed_numpy_author_freq[i]
            prob_val = word_count / total_word_freq
            dict_word_prob[word_key] = prob_val
        dict_author_cond_prob[author] = dict_word_prob

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
    tweet_list_bag = get_tweet_list_bag(list_tweets)
    dict_author_tgt = get_author_tgt_dict(list_tweets)
    tweet_np_matrix = get_tweet_list_frequencies_numpy_array(
        list_corpus, tweet_list_bag, dict_author_tgt
    )
    return [list_corpus, dict_author_tgt, tweet_np_matrix]


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

    list_corpus = list_data[0]  # list of class to tweets
    dict_author_tgt = list_data[1]  # map of author to random integer
    tweet_np_matrix = list_data[
        2
    ]  # list of list of author, word count, word count, ...

    dict_corpus_prob = get_corpus_prob_dict(
        list_corpus
    )  # map of word to prob entire words
    dict_author_prob = get_author_prob_dict(
        list_tweets
    )  # map of author probabilities in tweet data
    dict_cond_prob = get_cond_prob_dict(list_corpus, tweet_np_matrix, dict_author_tgt)

    data_set = [dict_corpus_prob, dict_author_prob, dict_cond_prob]
    return data_set


def predict(data_set, tweet):
    """
    Calculates and returns the Naive Bayesian score for TWEET based on DATASET
    :param data_set: a list consisting of three data:

    The first is a dictionary comprised of word to probability pairs.

    The second is a dictionary comprised of author to probability pairs.

    The third is a dictionary comprised of author to dictionary pairs. The
    dictionary value is comprised of word to probability pairs.

        example:

        { "donald" :  { "wall" : .23, "great" : .009},
          "hillary" : { "predator" : .32, "democrat" : .119} }

    :param tweet: a string representing a tweet by a candidate

    :return: list of classes that TWEET might belong to with decreasing
    probability
    """
    list_prob_of_tweet = []
    dict_corpus_prob = data_set[0]
    dict_author_prob = data_set[1]
    dict_cond_prob = data_set[2]
    list_token_tweet = tokenize_tweet(tweet)
    tweet_bag = make_tweet_bag(list_token_tweet)

    # does what hmap does, which is determine the probability of each possible outcome
    for author, prob in dict_author_prob.items():
        log_prob_author_score = 0.0
        dict_word_cond_prob = dict_cond_prob.get(
            author
        )  # The probability table of an author
        for word, count in tweet_bag.items():
            prob_word = dict_corpus_prob.get(
                word
            )  # The probablity of the word even occuring in the vocabulary given
            prob_word_author = dict_word_cond_prob.get(
                word
            )  # the probability of the word said by the author
            if prob_word_author > 0:
                log_prob_author_score += math.log(count * prob_word_author / prob_word)
        prob_author = dict_author_prob.get(author)
        log_prob_author_score = math.exp(log_prob_author_score + math.log(prob_author))
        result = (author, log_prob_author_score)
        list_prob_of_tweet.append(result)

    list_prob_of_tweet.sort(key=lambda tup: tup[1], reverse=True)
    return list_prob_of_tweet


def evaluation(document, test_data):
    """
    Computes the overall accuracy and the per class accuracy of classifier
    Precision - Out of all the answers by the classifier, which ones are correct?
    Recall - Out of all the actual correct answers, which ones are returned by the classifier?
    F1-Score - 2 * precision * reecall / (precision + recall)

    :param test_data:
    :param document: a csv file

    :return: a string showing a report of the classfier
    """
    data_set = train(document)
    list_test_data = read_test_data(test_data)
    list_truth = map(lambda x: x[0], list_test_data)
    list_prediction = []
    list_test_data_predict = map(lambda x: x[1], list_test_data)

    for test_tweet in list_test_data_predict:
        result = predict(data_set, test_tweet)
        prediction = result[0]
        list_prediction.append(prediction[0])

    report = classification_report(list_truth, list_prediction)
    return report


def test_scikit():
    list_tweets = read_data(FILE)
    scikit_args = get_scikit_fit_args(list_tweets)
    # print "Printing freq matrix"
    # print scikit_args[0]
    print(f"shape: {np.shape(scikit_args[0])}")
    # print "Printing target vector"
    # print scikit_args[1]
    print(f"shape: {np.shape(scikit_args[1])}")
    text_clf = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", GaussianNB()),
        ]
    )
    text_clf = text_clf.fit(scikit_args[0], scikit_args[1])
    with open(TEST_DATA, MODE) as csvfile:
        reader = csv.reader(csvfile, delimiter=DELIMITER, quotechar=QUOTECHAR)
        next(reader, None)
        test_data_list = map(lambda tweet: tweet[1], reader)
    predicted_target = text_clf.predict(test_data_list)
    score = np.mean(predicted_target == TEST_TARGET)

    # printing reports
    print(f"Score: {score}")
    print(
        metrics.classification_report(
            TEST_TARGET, predicted_target, target_names=TEST_TARGET
        )
    )
    print(f"Confusion matrix:\n")
    print(f"{metrics.confusion_matrix(TEST_TARGET, predicted_target)}")
    return 1


def test_train():
    data_set = train(FILE)
    dict_corpus_prob = data_set[0]
    dict_author_prob = data_set[1]
    dict_cond_prob = data_set[2]

    print("This is the corpus dictionary")
    for key, value in dict_corpus_prob.items():
        print(f"Key: {key}")
        print(f"Val: {value}")
        break

    print(f"This is the author prob dictionary")
    for key, value in dict_author_prob.items():
        print(f"Key: {key}")
        print(f"Val: {value}")

    print("This is the cond prob dictionary")
    for key, value in dict_cond_prob.items():
        print("\n", "The key is: ", key)
        for word, prob in value.items():
            print(f"Word: {word}")
            print(f"Prob: {prob}")
            break
    return 1


def tests():
    """
    Black and glass box testing for all methods
    :return: True
    """
    list_tweets = read_data(FILE)

    print(f"The total number of tweets: {len(list_tweets)}")

    for author, tweet in list_tweets:
        print(author)
        print(tweet)
        print("\n")
        break

    list_corpus = get_corpus_list(list_tweets)
    print(f"The total number of unique words: {len(list_corpus)}")
    for tup in list_corpus:
        print(tup[0])
        print(tup[1])
        print("\n")
        break

    list_tweet_bag = get_tweet_list_bag(list_tweets)
    print(f"The total number of tweets: {len(list_tweet_bag)}")
    for tup in list_tweet_bag:
        print(tup[0])
        print(tup[1])
        print("\n")
        break

    dict_auth_tgt = get_author_tgt_dict(list_tweets)
    for auth, tgt in dict_auth_tgt.items():
        print(auth)
        print(tgt)
        print("\n")

    tweet_np = get_tweet_list_frequencies_numpy_array(
        list_corpus, list_tweet_bag, dict_auth_tgt
    )
    numpy_shape = np.shape(tweet_np)
    width = numpy_shape[1]
    fit_x_arg = tweet_np[:, 1:width]
    fit_y_arg = tweet_np[:, 0]
    print("The numpy vector of target: ")
    print(fit_y_arg)
    print("\n", "The numpy array of freq count of all words: ")
    print(fit_x_arg)

    dict_corpus_prob = get_corpus_prob_dict(list_corpus)
    for key, value in dict_corpus_prob.items():
        print("\n")
        print("The word is: ", key)
        print("The prob is: ", value)
    totalprob = sum(dict_corpus_prob.values())
    print("\n", "The sum of all prob is: ", totalprob)

    dict_author_prob = get_author_prob_dict(list_tweets)
    for key, value in dict_author_prob.items():
        print("\n")
        print("The author is: ", key)
        print("The prob is: ", value)
    totalprob = sum(dict_author_prob.values())
    print("\n", "The sum of all prob is: ", totalprob)

    dict_cond_prob = get_cond_prob_dict(list_corpus, tweet_np, dict_auth_tgt)
    for key, value in dict_cond_prob.items():
        print("\n")
        print("The author is: ", key)
        for word, prob in value.items():
            print("The word is: ", word, "\n")
            print("The prob is: ", prob)
            break
        totalprob = sum(value.values())
        print("\n", "The sum of all prob is: ", totalprob)

    return True


def test_predict():
    data_set = train(FILE)
    result = predict(data_set, TEST_DONALD)
    print("\n", "Prediction from first test: ")
    for item in result:
        print("\n", item)
    result = predict(data_set, TEST_HILLARY)
    print("\n", "Prediction from second test: ")
    for item in result:
        print("\n", item, "\n")
    return 1


def test_evaluation():
    result = evaluation(FILE, TEST_DATA)
    print(result)
    return True


def main():
    # return test_evaluation()
    # return test_predict()
    # return test_train()
    # return test_scikit()
    return tests()


if __name__ == "__main__":
    main()
