
from collections import Counter, defaultdict
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

from datareaders import DataReader, KFoldDataReader

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

def tweet_features(tweet, bigrams=True):
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
  if bigrams:
    for tok1, tok2 in itertools.izip(tokens[:-1], tokens[1:]):
      if not ONLY_PUNCTUATION_RE.match(tok1) and not ONLY_PUNCTUATION_RE.match(tok2):
        if tok1 < tok2:
          yield "<2>{},{}</2>".format(tok1, tok2)
        else:
          yield "<2>{},{}</2>".format(tok2, tok1)
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

def bernoulli(training_data):
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

def frequencies(training_data):
  """
  Features and labels estimated using class frequencies
  """
  # estimate frequencies
  condfreqs = defaultdict(Counter)
  numtraining = 0
  labels = []
  for tweetinfo in training_data:
    label = tweetinfo["Answer"]
    for feat in tweet_features(tweetinfo, bigrams=False):
      condfreqs[label][feat] += 1
      numtraining += 1

  featureMap = {}
  labelMap = {}

  # normalize freqs to get probs
  for i, label in enumerate(condfreqs):
    denominator = float(sum(condfreqs[label].itervalues()))
    labelMap[label] = i
    featureMap[label] = i
    for feat, freq in condfreqs[label].iteritems():
      condfreqs[label][feat] /= denominator

  # convert to np.array
  features = np.zeros((numtraining, len(condfreqs)), dtype=float)
  labels = np.zeros((numtraining,), dtype=np.uint8)

  for i, tweetinfo in enumerate(training_data):
    label = tweetinfo["Answer"]
    labels[i] = labelMap[ label ]
    for feat in tweet_features(tweetinfo, bigrams=False):
      for label in labelMap:
        features[i][ featureMap[label] ] += condfreqs[label][feat]
    # features[i, :] /= np.sum(features[i, :])

  return (features, condfreqs, featureMap, labels, labelMap)

def m_randomforest():
  rf_learner = randomforest.rf_learner()
  return multi.one_against_one(rf_learner)

def m_svm():
  if ARGV.one_vs:
    svm_model = svm.svm_to_binary(svm.svm_raw())
  else:
    svm_model = multi.one_against_one(svm.svm_to_binary(svm.svm_raw()))
  # return milk.defaultclassifier(mode='slow', multi_strategy='1-vs-1')
  learner = milk.supervised.classifier.ctransforms(
    # remove nans
    supervised.normalise.chkfinite(),
    # normalize to [-1,1]
    supervised.normalise.interval_normalise(),
    # feature selection
    featureselection.featureselector(
       featureselection.linear_independent_features),
    # sda filter
    featureselection.sda_filter(),
    # same parameter range as 'medium'
    supervised.gridsearch(
      svm_model,
      params = {
        'C': 2.0 ** np.arange(-2, 4),
        'kernel': [ svm.rbf_kernel(2.0 ** i) for i in xrange(-4, 4) ]
      }
    )
  )
  return learner

models = {
  "randomforest": m_randomforest,
  "svm": m_svm
}

def train(training_data):
  """
  Trains a model, using bernoulli features
  """
  if ARGV.features ==  "bernoulli":
    features, featureMap, labels, labelMap = bernoulli(training_data)
  else:
    features, condfreqs, featureMap, labels, labelMap = frequencies(training_data)
  learner = models[ ARGV.model ]()
  if ARGV.one_vs:
    labels[ labels != labelMap[ ARGV.one_vs ] ] = 0
    labels[ labels == labelMap[ ARGV.one_vs ] ] = 1
  model = learner.train(features, labels)
  if ARGV.features ==  "bernoulli":
    return (model, featureMap, labelMap)
  else:
    return ((model, condfreqs), featureMap, labelMap)

def test(test_data, model, featureMap, labelMap):
  """
  Tests the model accuracy on test_data
  """
  numcorrect = 0
  numtotal = 0
  nummissing = 0
  if ARGV.features == "frequencies":
    model, condfreqs = model
  for tweetinfo in test_data:
    featuresFound = tweet_features(tweetinfo)
    features = np.zeros((len(featureMap), ), dtype=float)
    for feat in featuresFound:
      if ARGV.features == "frequencies":
        for label in labelMap:
          features[ featureMap[label] ] += condfreqs[label][feat]
      else:
        if feat in featureMap:
          features[ featureMap[feat] ] = 1
        else:
          nummissing += 1
    # features /= np.sum(features)
    guess = model.apply(features)
    if ARGV.one_vs:
      positive = labelMap[ ARGV.one_vs ]
      if guess != positive:
        if labelMap[ tweetinfo["Answer1"] ] != positive or labelMap[ tweetinfo["Answer2"] ] != positive:
          numcorrect += 1
      else:
        if labelMap[ tweetinfo["Answer1"] ] == positive or labelMap[ tweetinfo["Answer2"] ] == positive:
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
  features, featureMap, labels, labelMap = bernoulli(data.train())
  learner = models[ ARGV.model ]()
  model = learner.train(features, labels)
  cmatrix, names, predictions = milk.nfoldcrossvalidation(features, labels, nfolds=2, learner=learner, return_predictions=True)
  
  print cmatrix
  print "correct", cmatrix.trace()
  print "total", cmatrix.sum()
  print "accuracy", float(cmatrix.trace())/ cmatrix.sum()

def crossval_seq(data):
  allfolds_correct = 0
  allfolds_total = 0
  allfolds_missing = 0

  for fold in xrange(1, data.kfolds + 1):
    print "---* fold {} *---".format(fold)
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
  print "---* Overall results *---"
  print "Results:\n{} out of {} correct".format(allfolds_correct, allfolds_total)
  print "Accuracy {:.2f}".format(float(allfolds_correct) / allfolds_total)
  print "Missing features:\n{} out of {} missing".format(allfolds_missing, len(featureMap))

def predict():
  data = DataReader(ARGV.data)
  model_fname = "{}_model.pickle".format(ARGV.model)
  if ARGV.one_vs:
    model_fname = "{}_{}_model.pickle".format(ARGV.model, ARGV.one_vs)
  inp_fname = path.join("models", model_fname)
  print "---* Reading in from {} *---".format(inp_fname)
  with open(inp_fname, "rb") as inp:
    model, featureMap, labelMap = pickle.load(inp)
  if ARGV.features == "frequencies":
    model, condfreqs = model
  print "---* Predicting using {} *---".format(ARGV.model)
  if not path.exists("predictions"):
    os.mkdir("predictions")
  inp = path.basename(ARGV.data).split(".")[0]
  output_fname = path.join("predictions", "{}_{}.txt".format(ARGV.model, inp))
  if ARGV.one_vs:
    output_fname = path.join("predictions", "{}_{}_{}.txt".format(ARGV.model, ARGV.one_vs, inp))
  print "Writing preditions to: {}".format(output_fname)
  with open(output_fname, "w") as out:
    nummissing = 0
    invLabelMap = {}
    header = ["datetime", "label"] # "tweet"
    out.write("\t".join(header) + "\n")
    for label, label_id in labelMap.iteritems():
      invLabelMap[ label_id ] = label
    for tweetinfo in data:
      featuresFound = tweet_features(tweetinfo)
      features = np.zeros((len(featureMap), ), dtype=float)
      for feat in featuresFound:
        if ARGV.features == "frequencies":
          for label in labelMap:
            features[ featureMap[label] ] += condfreqs[label][feat]
        else:
          if feat in featureMap:
            features[ featureMap[feat] ] = 1
          else:
            nummissing += 1
      guess = model.apply(features)
      datestring = tweetinfo["Datetime"].strftime("%Y-%m-%d %H:%M:%S")
      out.write("{}\t{}\n".format( datestring, guess )) # tweetinfo["Tweet"]
  print "Number of new features: {}".format(nummissing)

def train_model():
  data = DataReader(ARGV.data, highp=True)
  print "---* Training {} model *---".format(ARGV.model)
  model, featureMap, labelMap = train(data)
  if not path.exists("models"):
    os.mkdir("models")
  model_fname = "{}_model.pickle".format(ARGV.model)
  if ARGV.one_vs:
    model_fname = "{}_{}_model.pickle".format(ARGV.model, ARGV.one_vs)
  output_fname = path.join("models", model_fname)
  print "Writing model to: {}".format(output_fname)
  with open(output_fname, "wb") as out:
    pickle.dump((model, featureMap, labelMap), out, pickle.HIGHEST_PROTOCOL)

def crossval():
  data = KFoldDataReader(ARGV.data, ARGV.k_folds, highp=True)
  print "---* KFold crossval for {} model *---".format(ARGV.model)
  if ARGV.parallel:
    crossval_parallel(data)
  else:
    crossval_seq(data)

def kmeans_summary():
  print "---* KMeans clustering *---"
  data = DataReader(ARGV.data)
  features, featureMap, labels, labelMap = bernoulli(data)
  # run kmeans
  k = len(labelMap)
  # pca_features, components = milk.unsupervised.pca(features)
  reduced_features = features
  cluster_ids, centroids = milk.unsupervised.repeated_kmeans(reduced_features, k, 3)
  # start outputing
  out_folder = "clusters"
  if not path.exists(out_folder):
    os.mkdir(out_folder)
  print "---* Results *---"
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
  # printing
  for i in xrange(k):
    if not ARGV.no_print:
      out_file = path.join(out_folder, "cluster_{}".format(i))
      print "Writing to: {}".format(out_file)
      with open(out_file, 'w') as out:
        for j, tweetinfo in enumerate(data):
          if cluster_ids[j] == i:
            out.write(tweetinfo["Tweet"] + "\n")
    if ARGV.plot:
      plt.plot(pca_features[cluster_ids == i, 1], pca_features[cluster_ids == i, 2], \
        colors[i] + marks[i])
  print Counter(cluster_ids)
  if ARGV.plot:
    print "Writing to: {}".format(path.join(out_folder, "plot.png"))
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

subparsers = parser.add_subparsers(title='Sub commands')

# debug
parser_debug = subparsers.add_parser('debug', help='Output features extracted')
parser_debug.add_argument("data", help="Input file")
parser_debug.add_argument("number", help="Number of tweets to output features of", type=int, default=10)
parser_debug.set_defaults(func=debug_features)

# cluster
parser_cluster = subparsers.add_parser('cluster', help='KMeans cluster data')
parser_cluster.add_argument("data", help="Input file")
parser_cluster.add_argument("-p", "--plot", help="Save plot of PCA reduced data", action="store_true")
parser_cluster.add_argument("-n", "--no-print", help="Include to avoid printing to output/", action="store_true")
parser_cluster.set_defaults(func=kmeans_summary)

# crossval
parser_crossval = subparsers.add_parser('crossval', help='Crossvalidation on data')
parser_crossval.add_argument("data", help="Input file")
parser_crossval.add_argument("model", help="Supervised model to use", choices=models.keys())
parser_crossval.add_argument("-p", "--parallel", help="Run KFold CV in Parallel", action="store_true")
parser_crossval.add_argument("-k", "--k-folds", help="K-Fold Cross Validation", type=int, default=10)
parser_crossval.add_argument("-f", "--features", choices=["bernoulli", "frequencies"], help="Features to extract", default="bernoulli")
parser_crossval.add_argument("-o", "--one-vs", choices=[ 'funny', 'none', 'afraid', 'angry', 'hopeful', 'sad', 'mocking', 'happy' ], help="One class to categorize on", default=None)
parser_crossval.set_defaults(func=crossval)

# train
parser_train = subparsers.add_parser('train', help='Train a model from the data')
parser_train.add_argument("data", help="Input file")
parser_train.add_argument("model", help="Supervised model to use", choices=models.keys())
parser_train.add_argument("-f", "--features", choices=["bernoulli", "frequencies"], help="Features to extract", default="bernoulli")
parser_train.add_argument("-o", "--one-vs", choices=[ 'funny', 'none', 'afraid', 'angry', 'hopeful', 'sad', 'mocking', 'happy' ], help="One class to categorize on", default=None)
parser_train.set_defaults(func=train_model)

# predict
parser_predict = subparsers.add_parser('predict', help='Predict labels for the data')
parser_predict.add_argument("data", help="Input file")
parser_predict.add_argument("model", help="Supervised model to use", choices=models.keys())
parser_predict.add_argument("-f", "--features", choices=["bernoulli", "frequencies"], help="Features to extract", default="bernoulli")
parser_predict.add_argument("-o", "--one-vs", choices=[ 'funny', 'none', 'afraid', 'angry', 'hopeful', 'sad', 'mocking', 'happy' ], help="One class to categorize on", default=None)
parser_predict.set_defaults(func=predict)

ARGV = parser.parse_args()

if __name__ == "__main__":
  ARGV.func()
