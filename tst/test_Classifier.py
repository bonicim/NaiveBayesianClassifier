import pytest
from os import path
from src.DataProcessor import DataProcessor
from src.Features import Features
from src.Classifier import Classifier


@pytest.fixture
def classifier_small():
    test_data_path = path.abspath(
        path.join(path.dirname(__file__), "..", "data", "tweet_test_data_small.csv")
    )
    classifier = Classifier(Features(DataProcessor(test_data_path)))
    classifier.train()

    return classifier


def test_small_data_should_predict_donald(classifier_small):
    tweet = "Crooked Hillary Clinton wants to flood our country with Syrian immigrants that we know little or nothing about. The danger is massive. NO!"
    expected = ("realDonaldTrump", 1.5)

    result = classifier_small.classify(tweet)

    assert result == expected


@pytest.fixture
def classifier_large_data_set():
    test_data_path = path.abspath(
        path.join(path.dirname(__file__), "..", "data", "tweets.csv")
    )
    classifier = Classifier(Features(DataProcessor(test_data_path)))
    classifier.train()

    return classifier


def test_large_data_should_predict_donald(classifier_large_data_set):
    tweet = "I refuse to call Megyn Kelly a bimbo, because that would not be politically correct. Instead I will only call her a lightweight reporter!"
    expected = ("realDonaldTrump", 9.128)

    result = classifier_large_data_set.classify(tweet)

    assert result == expected


def test_large_data_should_predict_hillary(classifier_large_data_set):
    tweet = "The boys are right. We need everyone's help to get the planet moving in the right direction. http://action1d.onedirectionmusic.com  #action1D"
    expected = ("HillaryClinton", 8.094)

    result = classifier_large_data_set.classify(tweet)

    assert result == expected


def test_evaluation(classifier_large_data_set):
    test_data = [
        (
            "HillaryClinton",
            "The question in this election: Who can put the plans into action that will make your life better?",
        ),
        (
            "realDonaldTrump",
            "It wasn't Matt Lauer that hurt Hillary last night. It was her very dumb answer about emails &amp; the veteran who said she should be in jail.",
        ),
    ]

    result = classifier_large_data_set.evaluation(test_data)

    print(f"\n\n{result}")
    assert result is not None
