import csv
import gzip
import re
import itertools
import os.path as path
from random import randint
from datetime import datetime

import feature_selection as fs

class Trainer:
  def __init__(self, reader, fold):
    self.reader = reader
    self.fold = fold
    self.elems = []
    if self.reader.partitioned:
      # already been through this
      for i, line in enumerate(self.reader.source.elems):
        if self.reader.fold_assignments[i] != self.fold:
          elems.append(line)
      return
    self.reader.numtotal = 0
    for i, line in enumerate(self.reader.source.elems):
      self.reader.numtotal += 1
      self.reader.fold_assignments.append( randint(1, self.reader.kfolds) )
      if self.reader.fold_assignments[i] != self.fold:
        self.elems.append(line)
    self.reader.partitioned = True

class KFoldDataReader:
  """
  Divides data into k folds
  """

  def __init__(self, inp_fname, kfolds=10, highp=False):
    self.source = DataReader(inp_fname, highp=highp)
    self.fold_assignments = []
    self.kfolds = kfolds
    self.numtotal = None
    self.featureMap = None
    self.labelMap2 = None
    self.partitioned = False

  def train(self, fold=1):
    return Trainer(self, fold)

  def test(self, fold=1):
    if not self.partitioned:
      raise RuntimeError("You must call .train() at least once before .test()")
    for i, line in enumerate(self.source.elems):
      if self.fold_assignments[i] == fold:
        yield line

  def all(self):
    for i, line in enumerate(self.source.elems):
      yield line

class DataReader:
  """
  Reads data, in a variety of formats
  on iterating, yields {{ dict with tweet info }}

  For gzipped data, file must be named "<fname>.format.gz"
  """

  def __init__(self, inp_fname, highp=False):
    self.input = inp_fname
    self.highp = highp
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
    if self.gzipped:
      f = gzip.open(self.input)
    else:
      f = open(self.input)
    reader = self.Reader(f)
    self.elems = []
    for info in reader.elems:
      if self.highp and not re.match(r"yes", info["Agreement"], re.I):
        continue
      info["Tweet"] = list(fs.tweet_features(info["Tweet"]))
      self.elems.append(info)
    f.close()


class TSVReader:
  """
  Tab-separated data reader
  """

  def __init__(self, source):
    self.source = source
    self.elems = []
    line = self.source.next().strip()
    header = re.split(r"\t", line)
    # header = ["tweet_id", "tweet", "label"]
    # header = ["author", "datetime", "tweet_id", "tweet"]
    for i, line in enumerate(self.source):
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
      if "Tweet" not in row:
        print "{}: {}".format(i+2, line)
        continue
      row["Agreement"] = "Yes"
      self.elems.append(row)

