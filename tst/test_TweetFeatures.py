import pytest
from os import path


@pytest.fixture
def tweet_probabilities():
    from src.TweetData import TweetData
    from src.TweetFeatures import TweetFeatures

    return TweetFeatures(
        TweetData(
            path.abspath(
                path.join(
                    path.dirname(__file__), "..", "data", "training_data_small.csv"
                )
            )
        )
    )


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

    prior_probs, vocab, cond_prob = tweet_probabilities.extract_features()

    assert prior_probs == expected
    assert vocab == expected_vocab
    assert cond_prob == expected_cond_prob
