import pytest
from os import path

test_data_path = path.abspath(
    path.join(path.dirname(__file__), "..", "data", "tweet_test_data_small.csv")
)


@pytest.fixture
def classifier():
    from src.DataProcessor import DataProcessor
    from src.Features import Features
    from src.Classifier import Classifier

    classifier = Classifier(Features(DataProcessor(test_data_path)))
    classifier.train()

    return classifier


def test_classify(classifier):
    tweet = "Crooked Hillary Clinton wants to flood our country with Syrian immigrants that we know little or nothing about. The danger is massive. NO!"

    result = classifier.classify(tweet)

    print(result)
    assert result is not None
