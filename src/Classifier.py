from collections import Counter
from math import log, exp
from sklearn.metrics import classification_report


class Classifier:
    def __init__(self, features):
        self._features = features
        self._prior_prob = None
        self._vocab = None
        self._cond_prob = None

    def train(self):
        self._prior_prob, self._vocab, self._cond_prob = (
            self._features.extract_features()
        )

    def classify(self, tweet):
        # turn tweet into bag of words
        words = Counter(tweet.split())
        hypothesis_prob = {}
        total_vocab = sum(self._vocab.values())

        # calculate the prob for each category
        for author, prob in self._prior_prob.items():
            hypothesis_prob[author] = 0.0
            author_vocab = self._cond_prob[author]
            total_author_vocab = sum(author_vocab.values())

            for word, count in words.items():
                if word not in self._vocab or len(word) <= 3:
                    continue
                prob_word = self._vocab.get(word) / total_vocab
                prob_given_category = author_vocab.get(word, 0) / total_author_vocab

                if prob_given_category > 0:
                    author_prob = hypothesis_prob[author]
                    author_prob += log(prob_given_category * count / prob_word)
                    hypothesis_prob[author] = author_prob

        hypothesis_prob = sorted(
            hypothesis_prob.items(), key=lambda t: t[1], reverse=True
        )
        hypothesis_prob = [(author, exp(prob)) for author, prob in hypothesis_prob]

        print(f"\n\n{hypothesis_prob}")

        return (hypothesis_prob[0][0], round(hypothesis_prob[0][1], 3))

    def evaluation(self, test_data):
        truth = [author for author, _ in test_data]
        prediction = []

        for _, tweet in test_data:
            result = self.classify(tweet)
            prediction.append(result[0])

        return classification_report(truth, prediction)
