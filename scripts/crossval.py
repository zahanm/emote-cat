from csv import DictReader
from random import randint

class KFoldData:

  def __init__(self, source, kfolds=10):
    self.source = source
    self.fold_assignments = []
    self.kfolds = kfolds
    self.numtotal = None
    self.featureMap = None
    self.labelMap = None
    self.partitioned = False

  def train(self, fold=1):
    with open(self.source) as f:
      reader = DictReader(f)
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
    with open(self.source) as f:
      reader = DictReader(f)
      for i, line in enumerate(reader):
        if self.fold_assignments[i] == fold:
          yield line

  def all(self):
    with open(self.source) as f:
      reader = DictReader(f)
      for i, line in enumerate(reader):
        yield line
