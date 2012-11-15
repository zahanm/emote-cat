
from csv import DictReader
from random import random

class KFoldData:

  def __init__(self, source):
    self.source = source
    self.test_indices = set()

  def train(self):
    with open(self.source) as f:
      reader = DictReader(f)
      for i, line in enumerate(reader):
        if random() < 0.9:
          yield line
        else:
          self.test_indices.add(i)

  def test(self):
    if len(self.testdata) == 0:
      raise RuntimeError("You must call .traindata() before .testdata()")
    with open(self.source) as f:
      reader = DictReader(f)
      for i, line in enumerate(reader):
        if i in self.test_indices:
          yield line
