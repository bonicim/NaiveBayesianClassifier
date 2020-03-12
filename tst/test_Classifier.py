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


def test_classify_small_model(classifier_small):
    tweet = "Crooked Hillary Clinton wants to flood our country with Syrian immigrants that we know little or nothing about. The danger is massive. NO!"

    result = classifier_small.classify(tweet)

    print(result)
    assert result is not None


@pytest.fixture
def classifier_large_data_set():
    test_data_path = path.abspath(
        path.join(path.dirname(__file__), "..", "data", "tweets.csv")
    )
    classifier = Classifier(Features(DataProcessor(test_data_path)))
    classifier.train()

    return classifier


def test_classify_large_data_set(classifier_large_data_set):
    tweet = "No legit reason @realDonaldTrump can't release returns while being audited, but if scared, release earlier returns no longer under audit."
    result = classifier_large_data_set.classify(tweet)

    print(result)
    assert result is not None
