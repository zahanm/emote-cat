
from glob import iglob as glob
from csv import DictReader
from collections import Counter

stoplist = frozenset("romney")
classes = Counter()
vocab = Counter()

def iterdata(source):
  with open(source) as f:
    reader = DictReader(f)
    for line in reader:
      yield line



def bagofwords():
  trainfile = "../../Tweet-Data/Romney-Labeled.csv"
  for data in iterdata(trainfile):
    words = data["Tweet"].split(r"\s+")
    for word in words:
      vocab[ word ] += 1
    classes[ data["Answer1"] ] += 1
    classes[ data["Answer2"] ] += 1
    classes[ data["Answer"] ] += 1
  print classes

if __name__ == "__main__":
  bagofwords()
