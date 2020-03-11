import pytest
from os import path

test_data_path = path.abspath(
    path.join(path.dirname(__file__), "..", "data", "tweet_test_data_small.csv")
)


@pytest.fixture
def data_processor():
    from src.DataProcessor import DataProcessor

    return DataProcessor(test_data_path)


def test_read(data_processor):
    expected = [
        (
            "HillaryClinton",
            "The question in this election: Who can put the plans into action that "
            "will make your life better? https://t.co/XreEY9OicG",
        ),
        (
            "realDonaldTrump",
            "Nothing on emails. Nothing on the corrupt Clinton Foundation. And "
            "nothing on #Benghazi. #Debates2016 #debatenight",
        ),
    ]

    actual = data_processor.read()

    assert actual == expected


def test_tokenize(data_processor):
    expected = [
        (
            "HillaryClinton",
            [
                "the",
                "question",
                "in",
                "this",
                "election",
                "who",
                "can",
                "put",
                "the",
                "plans",
                "into",
                "action",
                "that",
                "will",
                "make",
                "your",
                "life",
                "better",
                "httpstcoxreey9oicg",
            ],
        ),
        (
            "realDonaldTrump",
            [
                "nothing",
                "on",
                "emails",
                "nothing",
                "on",
                "the",
                "corrupt",
                "clinton",
                "foundation",
                "and",
                "nothing",
                "on",
                "benghazi",
                "debates2016",
                "debatenight",
            ],
        ),
    ]

    actual = data_processor.tokenize()

    assert actual == expected
