from csv import DictReader
from random import random

class KFoldData:

  def __init__(self, source, kfrac=0.1):
    self.source = source
    self.test_indices = set()
    self.kfrac = kfrac
    self.numtraining = None
    self.featureMap = None
    self.labelMap = None
    self.sets_partitioned = False

  def train(self):
    with open(self.source) as f:
      reader = DictReader(f)
      if self.sets_partitioned:
        # already been through this
        for i, line in enumerate(reader):
          if i not in self.test_indices:
            yield line
        return
      self.numtraining = 0
      for i, line in enumerate(reader):
        if random() >= self.kfrac:
          self.numtraining += 1
          yield line
        else:
          self.test_indices.add(i)
      self.sets_partitioned = True

  def test(self):
    if not self.sets_partitioned:
      raise RuntimeError("You must call .traindata() before .testdata()")
    with open(self.source) as f:
      reader = DictReader(f)
      for i, line in enumerate(reader):
        if i in self.test_indices:
          yield line

  def all(self):
    with open(self.source) as f:
      reader = DictReader(f)
      for i, line in enumerate(reader):
        yield line
