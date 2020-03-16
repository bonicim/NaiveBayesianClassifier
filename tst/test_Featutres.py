import pytest
from os import path

test_data_path = path.abspath(
    path.join(path.dirname(__file__), "..", "data", "tweet_test_data_small.csv")
)


@pytest.fixture
def tweet_probabilities():
    from src.TweetData import TweetData
    from src.TweetProbabilities import TweetProbabilities

    return TweetProbabilities(TweetData(test_data_path))


def test_extract(tweet_probabilities):
    expected = {
        "realDonaldTrump": -0.6931471805599453,
        "HillaryClinton": -0.6931471805599453,
    }
    expected_vocab = {
        "question": 1,
        "election": 1,
        "put": 1,
        "plans": 1,
        "action": 1,
        "make": 1,
        "life": 1,
        "better": 1,
        "httpstcoxreey9oicg": 1,
        "nothing": 4,
        "emails": 1,
        "corrupt": 1,
        "clinton": 1,
        "foundation": 1,
        "benghazi": 1,
        "debates2016": 1,
        "debatenight": 1,
    }
    expected_cond_prob = {
        "realDonaldTrump": {
            "nothing": 0.3,
            "emails": 0.1,
            "corrupt": 0.1,
            "clinton": 0.1,
            "foundation": 0.1,
            "benghazi": 0.1,
            "debates2016": 0.1,
            "debatenight": 0.1,
        },
        "HillaryClinton": {
            "question": 0.1,
            "election": 0.1,
            "put": 0.1,
            "plans": 0.1,
            "action": 0.1,
            "make": 0.1,
            "life": 0.1,
            "better": 0.1,
            "httpstcoxreey9oicg": 0.1,
            "nothing": 0.1,
        },
    }

    probs = tweet_probabilities.extract_features()

    prior_probs = probs[0]
    vocab = probs[1]
    cond_prob = probs[2]

    assert prior_probs == expected
    assert vocab == expected_vocab
    assert cond_prob == expected_cond_prob
