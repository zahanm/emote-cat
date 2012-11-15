
from collections import Counter
import itertools
import functools
import os
import os.path as path

import nltk
import emoticons
import numpy as np
import milk.unsupervised
import matplotlib.pyplot as plt

from crossval import KFoldData

PLOT_OUTPUT = False
porter = nltk.PorterStemmer()

stoplist = frozenset(["mitt", "romney", "barack", "obama", "the", "a", "is", "rt"])
def not_in_stoplist(t):
  return t not in stoplist

def to_lower(s):
  return s.lower()

def produce_data_maps(data):
  classes = Counter()
  vocab = Counter()
  numtraining = 0
  for tweetinfo in data.train():
    tokens = transform( tweetinfo["Tweet"] )
    for tok in tokens:
      vocab[ tok ] += 1
    classes[ tweetinfo["Answer1"] ] += 1
    numtraining += 1
  numtoks = len(vocab)
  featureMap = {}
  for j, tok in enumerate(vocab.iterkeys()):
    featureMap[tok] = j
  labelMap = {}
  for j, label in enumerate(classes.iterkeys()):
    labelMap[label] = j
  data.numtraining = numtraining
  data.featureMap = featureMap
  data.labelMap = labelMap

def extract_bernoulli(data):
  if data.numtraining == None or data.featureMap == None or data.labelMap == None:
    raise RuntimeError("Must run produce_data_maps(..) first")
  numtraining, featureMap, labelMap = data.numtraining, data.featureMap, data.labelMap
  numfeatures = len(featureMap)
  features = np.zeros((numtraining, numfeatures), dtype=np.uint8)
  labels = np.zeros((numtraining), dtype=np.uint8)
  for i, tweetinfo in enumerate(data.train()):
    tokens = transform( tweetinfo["Tweet"] )
    for tok in tokens:
      features[i, featureMap[tok]] = 1
    labels[i] = labelMap[ tweetinfo["Answer1"] ]
  return (features, labels)

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

def train_rf(data, features, labels):
  if data.numtraining == None or data.featureMap == None or data.labelMap == None:
    raise RuntimeError("Must run produce_data_maps(..) first")
  pass

def kmeans_summary(data, features, labels):
  if data.numtraining == None or data.featureMap == None or data.labelMap == None:
    raise RuntimeError("Must run produce_data_maps(..) first")
  # run kmeans
  k = len(data.labelMap)
  cluster_ids, centroids = milk.unsupervised.repeated_kmeans(features, k, 3)
  # start outputing
  out_folder = "output"
  if not path.exists(out_folder):
    os.mkdir(out_folder)
  # plot
  if PLOT_OUTPUT:
    transformed, components = milk.unsupervised.pca(features)
    colors = "bgrcbgrc"
    marks = "xxxxoooo"
    print [ np.min(transformed[:, 1]), np.max(transformed[:, 1]), \
      np.min(transformed[:, 2]), np.max(transformed[:, 2]) ]
    plt.axis([ np.min(transformed[:, 1]), np.max(transformed[:, 1]), \
      np.min(transformed[:, 2]), np.max(transformed[:, 2]) ])
  for i in xrange(k):
    out_file = path.join(out_folder, "cluster_{}".format(i))
    with open(out_file, 'w') as out:
      for j, tweetinfo in enumerate(data.train()):
        if cluster_ids[j] == i:
          out.write(tweetinfo["Tweet"] + "\n")
    if PLOT_OUTPUT:
      plt.plot(transformed[cluster_ids == i, 1], transformed[cluster_ids == i, 2], \
        colors[i] + marks[i])
  print Counter(cluster_ids)
  if PLOT_OUTPUT:
    plt.show()

if __name__ == "__main__":
  data = KFoldData("../Tweet-Data/Romney-Labeled.csv")
  produce_data_maps(data)
  features, labels = extract_bernoulli(data)
  kmeans_summary(data, features, labels)