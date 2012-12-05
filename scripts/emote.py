
from collections import Counter
import itertools
import functools
import re
import os
import os.path as path
import argparse
import cPickle as pickle

parser = argparse.ArgumentParser(description='Emotion analysis')
parser.add_argument("-p", "--plot", help="Include to show a plot", action="store_true")
parser.add_argument("-n", "--no-print", help="Include to avoid printing to output/", action="store_true")
parser.add_argument("-r", "--retrain", help="Retrain model", action="store_true")
parser.add_argument("-w", "--write", help="Writeout model", action="store_true")
parser.add_argument("-c", "--cluster", help="Run K-Means clustering", action="store_true")
parser.add_argument("-d", "--data", help="Dataset to use", choices=["romney", "tunisia", "obama", "topics"], default="romney")
parser.add_argument("-m", "--model", help="Model to train", choices=["randomforest", "svm"], default="svm")
ARGV = parser.parse_args()

import nltk
import emoticons
import numpy as np
from milk.supervised import randomforest
from milk.supervised import multi
import milk.unsupervised
if ARGV.plot:
  import matplotlib.pyplot as plt

from crossval import KFoldData

porter = nltk.PorterStemmer()

def tokenize(sentence):
  return re.split(r"\s+", sentence.strip())

stoplist = frozenset(["mitt", "romney", "barack", "obama", "the", "a", "is", "rt", "barackobama"])
def not_in_stoplist(t):
  return t not in stoplist

def to_lower(s):
  return s.lower()

def transform(text):
  """
  - lowercase
  - take out stoplist words
  - use porter stemmer
  """
  steps = [
    to_lower,
    tokenize,
    functools.partial(filter, not_in_stoplist),
    functools.partial(map, porter.stem)
  ]
  steps.reverse()
  current = text
  while len(steps) > 0:
    step = steps.pop()
    current = step(current)
  return current

def tweet_features(tweet):
  """
  Extracts a list of features for a given tweet

  Features:
  - singletons, bigrams
  - hashtags already included
  - emoticons
  - repeated punctuation
  - dialog RT @
  - sentiwordnet
  - slang / proper engish
  """
  rawtext = tweet["Tweet"]
  tokens = transform(rawtext)
  # singletons
  for tok in tokens:
    yield tok
  # bigrams
  for tok1, tok2 in itertools.izip(tokens[:-1], tokens[1:]):
    yield "<2>{},{}</2>".format(tok1, tok2)

def bernoulli_features(training_data):
  """
  Produces features and labels from training data, along with maps
  """
  featureMap = {}
  features = []
  numfeatures = 0
  labelMap = {}
  labels = []
  numlabels = 0
  numtraining = 0
  # produce featureMap and extract features together
  for tweetinfo in training_data:
    if not re.match(r"yes", tweetinfo["Agreement"], re.I):
      continue
    # add features to tweetvector
    tweetvector = [0] * numfeatures
    for feat in tweet_features(tweetinfo):
      if feat not in featureMap:
        featureMap[feat] = numfeatures
        numfeatures += 1
        tweetvector.append(0)
      tweetvector[ featureMap[feat] ] = 1
    numtraining += 1
    features.append(tweetvector)
    # store training label
    if tweetinfo["Answer"] not in labelMap:
      labelMap[ tweetinfo["Answer"] ] = numlabels
      numlabels += 1
    labels.append(labelMap[ tweetinfo["Answer"] ])
  # normalize lengths of feature vectors
  for i in xrange(len(features)):
    delta = numfeatures - len(features[i])
    if delta > 0:
      features[i].extend( [0] * delta )
  npfeatures = np.array(features, dtype=np.uint8)
  nplabels = np.array(labels, dtype=np.uint8)
  return (npfeatures, featureMap, nplabels, labelMap)

def train(training_data):
  """
  Trains a model, using bernoulli features
  """
  features, featureMap, labels, labelMap = bernoulli_features(training_data)
  learner = None
  if ARGV.model == "randomforest":
    rf_learner = randomforest.rf_learner()
    learner = multi.one_against_one(rf_learner)
  elif ARGV.model == "svm":
    svm_learner = milk.defaultclassifier()
    learner = multi.one_against_one(svm_learner)
  else:
    print "Invalid learning model: {}".format(ARGV.model)
    sys.exit(1)
  model = learner.train(features, labels)
  return (model, featureMap, labelMap)

def test(test_data, model, featureMap, labelMap):
  """
  Tests the model accuracy on test_data
  """
  numcorrect = 0
  numtotal = 0
  nummissing = 0
  for tweetinfo in test_data:
    featuresFound = tweet_features(tweetinfo)
    features = np.zeros((len(featureMap), ), dtype=np.uint8)
    for feat in featuresFound:
      if feat in featureMap:
        features[ featureMap[feat] ] = 1
      else:
        nummissing += 1
    guess = model.apply(features)
    if labelMap[ tweetinfo["Answer1"] ] == guess or labelMap[ tweetinfo["Answer2"] ] == guess:
      numcorrect += 1
    numtotal += 1
  return (numcorrect, numtotal, nummissing)

def kmeans_summary(data):
  features, featureMap, labels, labelMap = bernoulli_features(data.all())
  # run kmeans
  k = len(labelMap)
  # pca_features, components = milk.unsupervised.pca(features)
  reduced_features = features
  cluster_ids, centroids = milk.unsupervised.repeated_kmeans(reduced_features, k, 3)
  # start outputing
  out_folder = "output"
  if not path.exists(out_folder):
    os.mkdir(out_folder)
  # plot
  if ARGV.plot:
    colors = "bgrcbgrc"
    marks = "xxxxoooo"
    xmin = np.min(pca_features[:, 1])
    xmax = np.max(pca_features[:, 1])
    ymin = np.min(pca_features[:, 2])
    ymax = np.max(pca_features[:, 2])
    print [ xmin, xmax, ymin, ymax ]
    plt.axis([ xmin, xmax, ymin, ymax ])
  for i in xrange(k):
    if not ARGV.no_print:
      out_file = path.join(out_folder, "cluster_{}".format(i))
      with open(out_file, 'w') as out:
        for j, tweetinfo in enumerate(data.all()):
          if cluster_ids[j] == i:
            out.write(tweetinfo["Tweet"] + "\n")
    if ARGV.plot:
      plt.plot(pca_features[cluster_ids == i, 1], pca_features[cluster_ids == i, 2], \
        colors[i] + marks[i])
  print Counter(cluster_ids)
  if ARGV.plot:
    plt.show()

def classify_summary(data):
  if ARGV.retrain:
    print "Training {}".format(ARGV.model)
    training_data = data.train()
    model, featureMap, labelMap = train(training_data)
    if ARGV.write:
      with open("{}_model.pickle".format(ARGV.model), "wb") as out:
        pickle.dump((data, model, featureMap, labelMap), out, pickle.HIGHEST_PROTOCOL)
  else:
    print "Reading in {} model".format(ARGV.model)
    with open("{}_model.pickle".format(ARGV.model), "rb") as inp:
      data, model, featureMap, labelMap = pickle.load(inp)
  test_data = data.test()
  print "Testing {}".format(ARGV.model)
  numcorrect, numtotal, nummissing = test(test_data, model, featureMap, labelMap)
  print "Results:\n{} out of {} correct".format(numcorrect, numtotal)
  print "Accuracy {}".format(float(numcorrect) / numtotal)
  print "Features:\n{} out of {} missing".format(nummissing, len(featureMap))

def main():
  data = KFoldData( ARGV.data )
  if ARGV.cluster:
    kmeans_summary(data)
  else:
    classify_summary(data)

if __name__ == "__main__":
  main()
