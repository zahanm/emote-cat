
from csv import DictReader
from collections import Counter
import emoticons

from crossval import KFoldData

classes = Counter()
vocab = Counter()

stoplist = frozenset(["romney", "the", "a"])

def train(data):
  for tweetinfo in data.train():
    words = tweetinfo["Tweet"].split(r"\s+")
    for word in words:
      vocab[ word ] += 1
    classes[ tweetinfo["Answer1"] ] += 1
    classes[ tweetinfo["Answer2"] ] += 1
    classes[ tweetinfo["Answer"] ] += 1
  print classes

if __name__ == "__main__":
  data = KFoldData("../Tweet-Data/Romney-Labeled.csv")
  train(data)
