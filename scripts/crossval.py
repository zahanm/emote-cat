
from csv import DictReader
from random import randint
import re
import itertools

class KFoldData:

  def __init__(self, source, kfolds=10):
    if source == "romney":
      self.inpfile = "../Tweet-Data/Romney-Labeled.csv"
      self.reader = DictReader
    elif source == "tunisia":
      self.inpfile = "../Tweet-Data/Tunisia-Labeled.csv"
      self.reader = DictReader
    elif source == "obama":
      self.inpfile = "../Tweet-Data/Obama-Labeled.csv"
      self.reader = DictReader
    elif source == "topics":
      self.inpfile = "../Tweet-Data/topic-labeled-tweets.tsv"
      self.reader = TSVReader
    else:
      print "Invalid data source {}".format(source)
    self.fold_assignments = []
    self.kfolds = kfolds
    self.numtotal = None
    self.featureMap = None
    self.labelMap = None
    self.partitioned = False

  def train(self, fold=1):
    with open(self.inpfile) as f:
      reader = self.reader(f)
      if self.partitioned:
        # already been through this
        for i, line in enumerate(reader):
          if self.fold_assignments[i] != fold:
            yield line
        return
      self.numtotal = 0
      for i, line in enumerate(reader):
        self.numtotal += 1
        self.fold_assignments.append( randint(1, self.kfolds) )
        if self.fold_assignments[i] != fold:
          yield line
      self.partitioned = True

  def test(self, fold=1):
    if not self.partitioned:
      raise RuntimeError("You must call .traindata() before .testdata()")
    with open(self.inpfile) as f:
      reader = self.reader(f)
      for i, line in enumerate(reader):
        if self.fold_assignments[i] == fold:
          yield line

  def all(self):
    with open(self.inpfile) as f:
      reader = DictReader(f)
      for i, line in enumerate(reader):
        yield line

class TSVReader:

  def __init__(self, source):
    self.source = source

  def __iter__(self):
    line = self.source.next().strip()
    header = re.split(r"\s+", line)
    header = ["tweet_id", "tweet", "label"]
    for line in self.source:
      items = line.strip().split('\t')
      row = {}
      for key, val in itertools.izip(header, items):
        if key == 'tweet_id':
          row["TweetId"] = val
        elif key == "tweet":
          row["Tweet"] = val
        else:
          row["Answer"] = val
          row["Answer1"] = val
          row["Answer2"] = val
      row["Agreement"] = "Yes"
      yield row
