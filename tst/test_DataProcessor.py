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
            {
                "question": 1,
                "election": 1,
                "put": 1,
                "plans": 1,
                "action": 1,
                "make": 1,
                "life": 1,
                "better": 1,
                "httpstcoxreey9oicg": 1,
                "nothing": 1,
            },
        ),
        (
            "realDonaldTrump",
            {
                "nothing": 3,
                "emails": 1,
                "corrupt": 1,
                "clinton": 1,
                "foundation": 1,
                "benghazi": 1,
                "debates2016": 1,
                "debatenight": 1,
            },
        ),
    ]

    actual = data_processor.process()

    assert actual == expected
