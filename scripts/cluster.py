
from collections import Counter
import itertools
import functools

import nltk
import emoticons
import numpy as np

from crossval import KFoldData

classes = Counter()
vocab = Counter()

stoplist = frozenset(["romney", "obama", "the", "a", "is"])
def not_in_stoplist(t):
  return t not in stoplist

def to_lower(s):
  return s.lower()

def features_bernoulli(data):
  porter = nltk.PorterStemmer()
  numtraining = 0
  for tweetinfo in data.train():
    tokens = transform( tweetinfo["Tweet"] )
    for tok in tokens:
      vocab[ tok ] += 1
    classes[ tweetinfo["Answer1"] ] += 1
    numtraining += 1
  numtoks = len(vocab)
  featureMap = {}
  for j, tok in enumerate(vocab.iterkeys()):
    featureMap[tok] = j
  labelMap = {}
  for j, label in enumerate(classes.iterkeys()):
    labelMap[label] = j
  features = np.zeros((numtraining, numtoks), dtype=np.bool)
  labels = np.zeros((numtraining), dtype=np.uint8)
  for i, tweetinfo in enumerate(data.train()):
    tokens = transform( tweetinfo["Tweet"] )
    for tok in tokens:
      features[i, featureMap[tok]] = True
    labels[i] = labelMap[ tweetinfo["Answer1"] ]
  return (features, featureMap, labels, labelMap)

porter = nltk.PorterStemmer()
def transform(text):
  """
  - lowercase
  - take out stoplist words
  - use porter stemmer
  """
  steps = [
    to_lower,
    nltk.word_tokenize,
    functools.partial(filter, not_in_stoplist),
    functools.partial(map, porter.stem)
  ]
  steps.reverse()
  current = text
  while len(steps) > 0:
    step = steps.pop()
    current = step(current)
  return current

if __name__ == "__main__":
  data = KFoldData("../Tweet-Data/Romney-Labeled.csv")
  features, featureMap, labels, labelMap = features_bernoulli(data)
