
from collections import Counter
import itertools
import functools
import re
import os
import os.path as path
import argparse
import cPickle as pickle

import nltk
import emoticons
import twokenize
import numpy as np
import milk.supervised as supervised
from milk.supervised import svm
from milk.supervised import randomforest
from milk.supervised import multi
from milk.supervised import featureselection
import milk.unsupervised

from crossval import DataReader, KFoldDataReader

porter = nltk.PorterStemmer()

stoplist = frozenset(["mitt", "romney", "barack", "obama", "the", "a", "is", "rt", "barackobama"])
def not_in_stoplist(t):
  return t not in stoplist

def to_lower(s):
  return s.lower()

# regular expressions for feature detection
ONLY_PUNCTUATION_RE = re.compile(r"^[\.,!?\-+;:\"'\s]+$")
REPEATED_PUNCTUATION_RE = re.compile(r"[\.!?]{2,}")
DIALOG_RE = re.compile(r"RT\s+|@\w+")
ALL_CAPS_RE = re.compile(r"[^\w@][A-Z]{2,}[\W]") # I know, the irony!

def transform(text):
  """
  - lowercase
  - take out stoplist words
  - use porter stemmer
  """
  steps = [
    to_lower,
    twokenize.tokenize, # nltk.word_tokenize,
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
  - all caps
  - dialog RT @
  - sentiwordnet
  - slang / proper engish
  """
  rawtext = tweet["Tweet"]
  tokens = transform(rawtext)
  # singletons
  for tok in tokens:
    if not ONLY_PUNCTUATION_RE.match(tok):
      yield tok
  # bigrams
  for tok1, tok2 in itertools.izip(tokens[:-1], tokens[1:]):
    if not ONLY_PUNCTUATION_RE.match(tok1) and not ONLY_PUNCTUATION_RE.match(tok2):
      yield "<2>{},{}</2>".format(tok1, tok2)
  # emoticons
  for emoticon in emoticons.analyze_tweet(rawtext):
    yield "<e>{}</e>".format(emoticon)
  # repeated punctuation
  if REPEATED_PUNCTUATION_RE.search(rawtext):
    yield "<rp>!</rp>"
  # dialog
  if DIALOG_RE.search(rawtext):
    yield "<d>!</d>"
  # all caps
  if ALL_CAPS_RE.search(rawtext):
    yield "<ac>!</ac>"

def bernoulli_features(training_data, highp=True):
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
    if highp and not re.match(r"yes", tweetinfo["Agreement"], re.I):
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

def m_randomforest():
  rf_learner = randomforest.rf_learner()
  return multi.one_against_one(rf_learner)

def m_svm():
  # return milk.defaultclassifier(mode='slow', multi_strategy='1-vs-1')
  return milk.supervised.classifier.ctransforms(
    supervised.normalise.chkfinite(),
    supervised.normalise.interval_normalise(),
    # no feature selection for now
    # featureselection.featureselector(
    #   featureselection.linear_independent_features),
    # featureselection.sda_filter(),
    # --
    # same parameter range as 'medium'
    supervised.gridsearch(
      multi.one_against_one(svm.svm_to_binary(svm.svm_raw())),
      params = {
        'C': 2.0 ** np.arange(-2, 4),
        'kernel': [ svm.rbf_kernel(2.0 ** i) for i in xrange(-4, 4) ]
      }
    )
  )

models = {
  "randomforest": m_randomforest,
  "svm": m_svm
}

def train(training_data):
  """
  Trains a model, using bernoulli features
  """
  features, featureMap, labels, labelMap = bernoulli_features(training_data)
  learner = models[ ARGV.model ]()
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

def crossval_parallel(data):

  from milk.ext.jugparallel import nfoldcrossvalidation

  # Import the parallel module
  from milk.utils import parallel

  # For this example, we rely on milksets
  from milksets.wine import load

  # Use all available processors
  parallel.set_max_processors()

  # Load the data
  features, featureMap, labels, labelMap = bernoulli_features(data.train())
  learner = models[ ARGV.model ]()
  model = learner.train(features, labels)
  cmatrix = nfoldcrossvalidation(features, labels, nfolds=2, learner=learner, return_predictions=True)
  print cmatrix
  print names
  print predictions

def crossval_seq(data):
  print "*** {} ***".format(ARGV.model)
  allfolds_correct = 0
  allfolds_total = 0
  allfolds_missing = 0

  for fold in xrange(1, data.kfolds + 1):
    print "--- fold {} ---".format(fold)
    print "training.."
    training_data = data.train(fold)
    model, featureMap, labelMap = train(training_data)
    print "testing.."
    test_data = data.test(fold)
    numcorrect, numtotal, nummissing = test(test_data, model, featureMap, labelMap)
    print "Results:\n{} out of {} correct".format(numcorrect, numtotal)
    print "Accuracy {:.2f}".format(float(numcorrect) / numtotal)
    allfolds_correct += numcorrect
    allfolds_total += numtotal
    allfolds_missing += nummissing
  print "--- Overall results ---"
  print "Results:\n{} out of {} correct".format(allfolds_correct, allfolds_total)
  print "Accuracy {:.2f}".format(float(allfolds_correct) / allfolds_total)
  print "Missing features:\n{} out of {} missing".format(allfolds_missing, len(featureMap))
  if ARGV.write:
    with open("{}_model.pickle".format(ARGV.model), "wb") as out:
      pickle.dump((data, model, featureMap, labelMap), out, pickle.HIGHEST_PROTOCOL)

def predict():
  data = open(ARGV.data)
  print "Reading in {} model".format(ARGV.model)
  with open("{}_model.pickle".format(ARGV.model), "rb") as inp:
    data, model, featureMap, labelMap = pickle.load(inp)
  test_data = data.test()
  print "Testing {}".format(ARGV.model)
  numcorrect, numtotal, nummissing = test(test_data, model, featureMap, labelMap)
  print "Results:\n{} out of {} correct".format(numcorrect, numtotal)
  print "Accuracy {}".format(float(numcorrect) / numtotal)
  print "Features:\n{} out of {} missing".format(nummissing, len(featureMap))

def train_model():
  pass

def crossval():
  data = KFoldDataReader(ARGV.data, ARGV.k_folds)
  if ARGV.parallel:
    crossval_parallel(data)
  else:
    crossval_seq(data)

def kmeans_summary():
  data = DataReader(ARGV.data)
  features, featureMap, labels, labelMap = bernoulli_features(data, highp=False)
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
    import matplotlib.pyplot as plt
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
        for j, tweetinfo in enumerate(data):
          if cluster_ids[j] == i:
            out.write(tweetinfo["Tweet"] + "\n")
    if ARGV.plot:
      plt.plot(pca_features[cluster_ids == i, 1], pca_features[cluster_ids == i, 2], \
        colors[i] + marks[i])
  print Counter(cluster_ids)
  if ARGV.plot:
    plt.savefig(path.join(out_folder, "plot.png"))

def debug_features():
  data = DataReader(ARGV.data)
  for i, tweet in enumerate(data):
    if i > ARGV.number:
      break
    features = [ feat for feat in tweet_features(tweet) ]
    print "tweet: " + tweet["Tweet"]
    print "features: " + str(features)
    print

parser = argparse.ArgumentParser(description='Emotion analysis')
parser.add_argument("-v", "--verbose", help="Print debug information", action="store_true")
parser.add_argument("data", help="Input file for command")

subparsers = parser.add_subparsers(title='Sub commands')

# debug
parser_debug = subparsers.add_parser('debug', help='Output features extracted')
parser_debug.add_argument("number", help="Number of tweets to output features of", type=int, default=10)
parser_debug.set_defaults(func=debug_features)

# cluster
parser_cluster = subparsers.add_parser('cluster', help='KMeans cluster data')
parser_cluster.add_argument("-p", "--plot", help="Save plot of PCA reduced data", action="store_true")
parser_cluster.add_argument("-n", "--no-print", help="Include to avoid printing to output/", action="store_true")
parser_cluster.set_defaults(func=kmeans_summary)

# crossval
parser_crossval = subparsers.add_parser('crossval', help='Crossvalidation on data')
parser_crossval.add_argument("model", help="Supervised model to use", choices=models.keys())
parser_crossval.add_argument("-p", "--parallel", help="Run KFold CV in Parallel", action="store_true")
parser_crossval.add_argument("-k", "--k-folds", help="K-Fold Cross Validation", type=int, default=10)
parser_crossval.set_defaults(func=crossval)

# train
parser_train = subparsers.add_parser('train', help='Train a model from the data')
parser_train.add_argument("model", help="Supervised model to use", choices=models.keys())
parser_train.set_defaults(func=train_model)

# predict
parser_predict = subparsers.add_parser('predict', help='Predict labels for the data')
parser_predict.add_argument("model", help="Supervised model to use", choices=models.keys())
parser_predict.set_defaults(func=predict)

ARGV = parser.parse_args()

if __name__ == "__main__":
  ARGV.func()
