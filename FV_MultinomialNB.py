from utils import read_ov, remove_punctuation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import enum
from math import log10
import copy

tweet_id_index = 0
tweet_index = 1
label_index = 2

class q1_classification(enum.Enum):
  YES = 'yes'
  NO = 'no'


class FV_MultinomialNB:
  def __init__(self):
    self.alpha = 0.01
    self.smoothing = 0.01
    self.number_of_good_tweets = 0
    self.number_of_bad_tweets = 0
    self.number_of_total_tweets = 0
    self.training_data = None
    self.good_words = dict()
    self.bad_words = dict()
    self.good_word_likelihoods = dict()
    self.bad_word_likelihoods = dict()
    self.vocab_length = 0
    return

  def fit(self, filename: str):
    ov = read_ov(filename)
    vectorizer = CountVectorizer(lowercase=True)
    rows = [row for row in ov.to_numpy()]

    self.training_data = rows

    tweet_ids = [row[tweet_id_index] for row in rows]
    tweets = [row[tweet_index].lower() for row in rows]
    q1_labels = [row[label_index] for row in rows]

    OV_tweets = vectorizer.fit_transform(tweets)

    # Set number of 'yes' and 'no' labels
    for label in q1_labels:
      if label == q1_classification.YES.value:
        self.number_of_good_tweets = self.number_of_good_tweets + 1
      else:
        self.number_of_bad_tweets = self.number_of_bad_tweets + 1
    #

    # define number of words in good dict and bad dict
    for document in self.training_data:
      for word in remove_punctuation(document[tweet_index].lower()).split():
        if document[label_index] == q1_classification.YES.value:
          if word in self.good_words:
            self.good_words[word] = self.good_words[word] + 1
          else:
            self.good_words[word] = 1
        else:
          if word in self.bad_words:
            self.bad_words[word] = self.bad_words[word] + 1
          else:
            self.bad_words[word] = 1
    
    # REMOVE ALL INSTANCES WITH ONLY ONE WORDS
    temp = copy.deepcopy(self.good_words)
    for word, word_count in temp.items():
      if word_count == 1:
        del self.good_words[word]

    temp = copy.deepcopy(self.bad_words)
    for word, word_count in temp.items():
      if word_count == 1:
        del self.bad_words[word]
      


    # set vocab length (merge both good and bad dicts and compute number of keys)
    self.vocab_length = len({**self.good_words, **self.bad_words}.keys())
    
    # define probabilities of words being in factual or non-factual claim
    for word, word_count in self.good_words.items():
      likelihood = float(word_count + self.smoothing) / (self.number_of_good_tweets + self.smoothing * self.vocab_length)
      self.good_word_likelihoods[word] = likelihood

    for word, word_count in self.bad_words.items():
      likelihood = float(word_count + self.smoothing) / (self.number_of_bad_tweets + self.smoothing * self.vocab_length)
      self.bad_word_likelihoods[word] = likelihood

    return

  def predict(self, filename):
    print
    results = []

    ov = read_ov(filename, header=None)
    vectorizer = CountVectorizer(lowercase=True)
    rows = [row for row in ov.to_numpy()]

    for row in rows:
      good_score = log10(self.number_of_good_tweets / self.vocab_length) + sum([log10(self.good_word_likelihoods.get(word, 1)) for word in row[tweet_index]])
      bad_score = log10(self.number_of_bad_tweets / self.vocab_length) + sum([log10(self.bad_word_likelihoods.get(word, 1)) for word in row[tweet_index]])

      # print(good_score, bad_score)

      good = good_score >= bad_score

      results.append({"tweet_id": row[tweet_id_index], "class": q1_classification.YES.value if good else q1_classification.NO.value, "score": good_score})

    return results