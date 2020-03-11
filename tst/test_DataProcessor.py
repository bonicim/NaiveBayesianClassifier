import pytest
from os import path

test_data_path = path.abspath(
    path.join(path.dirname(__file__), "..", "data", "tweet_test_data_small.csv")
)


@pytest.fixture
def data_processor():
    from src.DataProcessor import DataProcessor

    return DataProcessor(test_data_path)


def test_process(data_processor):
    expected = [
        (
            "HillaryClinton",
            [
                "question",
                "election",
                "put",
                "plans",
                "action",
                "make",
                "life",
                "better",
                "httpstcoxreey9oicg",
            ],
        ),
        (
            "realDonaldTrump",
            [
                "nothing",
                "emails",
                "nothing",
                "corrupt",
                "clinton",
                "foundation",
                "nothing",
                "benghazi",
                "debates2016",
                "debatenight",
            ],
        ),
    ]

    actual = data_processor.process()

    assert actual == expected
