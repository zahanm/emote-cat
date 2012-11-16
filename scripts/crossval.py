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

  def train(self):
    with open(self.source) as f:
      reader = DictReader(f)
      self.numtraining = 0
      for i, line in enumerate(reader):
        if len(self.test_indices) == 0:
          if random() >= self.kfrac:
            self.numtraining += 1
            yield line
          else:
            self.test_indices.add(i)
        else:
          if i not in self.test_indices:
            yield line

  def test(self):
    if len(self.test_indices) == 0:
      raise RuntimeError("You must call .traindata() before .testdata()")
    with open(self.source) as f:
      reader = DictReader(f)
      for i, line in enumerate(reader):
        if i in self.test_indices:
          yield line
