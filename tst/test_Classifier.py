import pytest
from os import path
from src.TweetData import TweetData
from src.TweetProbabilities import TweetProbabilities
from src.Tweet import Tweet
import pprint


@pytest.fixture
def classifier_small():
    test_data_path = path.abspath(
        path.join(path.dirname(__file__), "..", "data", "training_data_small.csv")
    )
    classifier = Tweet(TweetProbabilities(TweetData(test_data_path)))
    classifier.train()

    return classifier


def test_small_data_should_predict_donald(classifier_small):
    tweet = "Crooked Hillary Clinton wants to flood our country with Syrian immigrants that we know little or nothing about. The danger is massive. NO!"
    expected = ["realDonaldTrump", "HillaryClinton"]

    predictions = classifier_small.classify(tweet)
    assert get_authors(predictions) == expected


@pytest.fixture
def classifier_large_data_set():
    train_data_path = path.abspath(
        path.join(path.dirname(__file__), "..", "data", "training_data.csv")
    )
    classifier = Tweet(TweetProbabilities(TweetData(train_data_path)))
    classifier.train()

    return classifier


@pytest.fixture
def tweet_testing_data_input():
    return TweetData(
        path.abspath(
            path.join(path.dirname(__file__), "..", "data", "testing_data.csv")
        )
    )


def test_large_data_should_predict_donald(classifier_large_data_set):
    tweet = "I refuse to call Megyn Kelly a bimbo, because that would not be politically correct. Instead I will only call her a lightweight reporter!"
    expected = ["realDonaldTrump", "HillaryClinton"]

    predictions = classifier_large_data_set.classify(tweet)

    assert get_authors(predictions) == expected


def test_large_data_should_predict_hillary(classifier_large_data_set):
    tweet = "The boys are right. We need everyone's help to get the planet moving in the right direction. http://action1d.onedirectionmusic.com  #action1D"
    expected = ["HillaryClinton", "realDonaldTrump"]

    predictions = classifier_large_data_set.classify(tweet)

    assert get_authors(predictions) == expected


def test_large_data_predict_given_list_of_tests(
    classifier_large_data_set, tweet_testing_data_input
):

    predictions = classifier_large_data_set.classify_collection_tweets(
        tweet_testing_data_input
    )

    assert len(predictions) == len(tweet_testing_data_input.generate_authors())


def get_authors(predictions):
    return [author for author, _ in predictions]
