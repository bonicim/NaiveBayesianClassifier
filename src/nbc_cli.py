#!/usr/bin/python

from .__init__ import __version__
from .TweetClassifier import TweetClassifier
from .TweetFeatures import TweetFeatures
from .TweetData import TweetData
import argparse
from os import path


def main():
    def train(args):
        classifier = TweetClassifier(TweetFeatures(TweetData(args.filepath)))
        classifier.train()

    def classify(args):
        classifier = TweetClassifier(TweetFeatures(TweetData(args.filepath)))
        classifier.train()
        print(classifier.classify(args.tweet))

    def no_command(args):
        if args.version:
            print("naive-bayesian-classifier-for-tweets", __version__)
        else:
            parser.print_help()

    # top level parser
    parser = argparse.ArgumentParser(
        description="Naive Bayesian Classifier that classifies tweets"
    )
    parser.set_defaults(command=no_command)
    parser.add_argument(
        "-v", "--version", action="store_true", help="show version and exit"
    )
    subparsers = parser.add_subparsers(dest="subparser_name")

    # subparser for training the model
    parser_train_model = subparsers.add_parser("train", help="train help")
    parser_train_model.add_argument("filepath", help="must be absolute path")
    parser_train_model.set_defaults(command=train)

    # subparser for classifying a tweet
    parser_classify = subparsers.add_parser("classify", help="classify help")
    parser_classify.add_argument("filepath", help="must be absolute path")
    parser_classify.add_argument("tweet", help="must be 140 chars max")
    parser_classify.set_defaults(command=classify)

    args = parser.parse_args()
    args.command(args)
