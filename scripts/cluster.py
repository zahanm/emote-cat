
from collections import Counter
import itertools
import functools

import nltk
import emoticons

from crossval import KFoldData

classes = Counter()
vocab = Counter()

stoplist = frozenset(["romney", "obama", "the", "a", "is"])
def not_in_stoplist(t):
  return t not in stoplist

def to_lower(s):
  return s.lower()

def train(data):
  porter = nltk.PorterStemmer()
  for tweetinfo in data.train():
    tokens = transform( tweetinfo["Tweet"] )
    for tok in tokens:
      vocab[ tok ] += 1
    classes[ tweetinfo["Answer1"] ] += 1
    classes[ tweetinfo["Answer2"] ] += 1
    classes[ tweetinfo["Answer"] ] += 1
  print classes

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
  train(data)
