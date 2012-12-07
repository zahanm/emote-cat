
import csv
import gzip
import re
import itertools
import os.path as path
from random import randint
from datetime import datetime

class KFoldDataReader:
  """
  Divides data into k folds
  """

  def __init__(self, inp_fname, kfolds=10):
    self.source = DataReader(inp_fname)
    self.fold_assignments = []
    self.kfolds = kfolds
    self.numtotal = None
    self.featureMap = None
    self.labelMap = None
    self.partitioned = False

  def train(self, highp=True, fold=1):
    source = self.source.all(highp=highp)
    if self.partitioned:
      # already been through this
      for i, line in enumerate(source):
        if self.fold_assignments[i] != fold:
          yield line
      return
    self.numtotal = 0
    for i, line in enumerate(source):
      self.numtotal += 1
      self.fold_assignments.append( randint(1, self.kfolds) )
      if self.fold_assignments[i] != fold:
        yield line
    self.partitioned = True

  def test(self, highp=False, fold=1):
    source = self.source.all(highp=highp)
    if not self.partitioned:
      raise RuntimeError("You must call .train() at least once before .test()")
    for i, line in enumerate(self.source):
      if self.fold_assignments[i] == fold:
        yield line

  def all(self, highp=False):
    source = self.source.all(highp=highp)
    for i, line in enumerate(source):
      yield line

class DataReader:
  """
  Reads data, in a variety of formats
  on iterating, yields {{ dict with tweet info }}

  For gzipped data, file must be named "<fname>.format.gz"
  """

  def __init__(self, inp_fname):
    self.input = inp_fname
    ext = path.splitext(inp_fname)[1]
    # gzip
    if ext == '.gz':
      self.gzipped = True
      ext = '.' + path.splitext(inp_fname)[0].split('.')[-1]
    else:
      self.gzipped = False
    readers = {
      '.csv': csv.DictReader,
      '.tsv': TSVReader
    }
    if ext in readers:
      self.Reader = readers[ext]
    else:
      raise RuntimeError("Unsupported input format")

  def all(self, highp=False):
    if self.gzipped:
      f = gzip.open(self.input)
    else:
      f = open(self.input)
    reader = self.Reader(f)
    for info in reader:
      if highp and not re.match(r"yes", info["Agreement"], re.I):
        continue
      yield info
    f.close()

class TSVReader:
  """
  Tab-separated data reader
  """

  def __init__(self, source):
    self.source = source

  def __iter__(self):
    line = self.source.next().strip()
    header = re.split(r"\t", line)
    # header = ["tweet_id", "tweet", "label"]
    # header = ["author", "datetime", "tweet_id", "tweet"]
    for line in self.source:
      items = line.strip().split('\t')
      row = {}
      for key, val in itertools.izip(header, items):
        if re.match(r"tweet_?id", key, re.I):
          row["TweetId"] = val
        elif re.match(r"tweet|text", key, re.I):
          row["Tweet"] = val
        elif re.match(r"label|class", key, re.I):
          row["Answer"] = val
          row["Answer1"] = val
          row["Answer2"] = val
        elif re.match(r"(date|time)+", key, re.I):
          row["Datetime"] = datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
        elif re.match(r"author|tweeter", key, re.I):
          row["Author"] = val
        else:
          print "WARN: Are you sure that the datafile has a header line?"
      row["Agreement"] = "Yes"
      yield row
