import numpy as np
from collections import Counter, defaultdict
import nltk
import twokenize
import functools
import re
import itertools
import emoticons
import math

porter = nltk.PorterStemmer()

stoplist = frozenset(["mitt", "romney", "barack", "obama", "the", "a", "is", "rt", "barackobama", "and", "for", "to", "at", "i", "that", "you", "of", "be", "it", "on"])

# regular expressions for feature detection
ONLY_PUNCTUATION_RE = re.compile(r"^[\.,!?\-+;:\"'\s]+$")
REPEATED_PUNCTUATION_RE = re.compile(r"[\.!?]{2,}")
DIALOG_RE = re.compile(r"RT\s+|@\w+")
ALL_CAPS_RE = re.compile(r"[^\w@][A-Z]{2,}[\W]") # I know, the irony!


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
  rawtext = tweet
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
  labelMap = Counter()
  labels = []
  numlabels = 0
  numtraining = 0


  """ First, see which features are significant """
  feature_threshold = 1
  feature_counter = Counter()
  for tweetinfo in training_data:
    for feat in tweetinfo["Features"]:
      feature_counter[feat] += 1

  # produce featureMap and extract features together
  for tweetinfo in training_data:
    # add features to tweetvector
    tweetvector = [0] * numfeatures
    for feat in tweetinfo["Features"]:
      if feat not in featureMap:
        if feature_counter[feat] < feature_threshold: continue
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
    for feat in tweetinfo["Features"]:
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
    for feat in tweetinfo["Features"]:
      for label in labelMap:
        features[i][ featureMap[label] ] += condfreqs[label][feat]
    # features[i, :] /= np.sum(features[i, :])

  return (features, condfreqs, featureMap, labels, labelMap)


def get_all_counts(training_data):
  feature_ctr = Counter()
  label_ctr = Counter()
  #count features within each label
  joint_ctr = defaultdict(Counter)
  total_tweets = 0
  for tweetinfo in training_data:
    total_tweets += 1
    label = tweetinfo["Answer"]
    label_ctr[label] += 1
    #Count features at most once
    for feat in set(tweetinfo["Features"]):
      feature_ctr[feat] += 1
      joint_ctr[label][feat] += 1
  return feature_ctr, label_ctr, joint_ctr, total_tweets

def mutualinfo(training_data):
  threshold = 0.01
  return run_fs(training_data, threshold)


def run_fs(training_data, threshold=0, chi=False):

  """ We need p(word & label), p(word), and p(label) """

  """ Get counts of all features and classes """
  feature_ctr, label_ctr, joint_ctr, total = get_all_counts(training_data)
  good_features = set()

  label_values = list(label_ctr.keys())
  label_map = dict(zip(label_values, range(len(label_values))))
  labels = []

  """ Now calculate all of the scores """
  mutual_info_scores = defaultdict(dict)  
  for feat in feature_ctr.keys():
    for label in label_values:
      #skip anything that doesn't occur
      if label_ctr[label]*feature_ctr[feat]*joint_ctr[label][feat] == 0:
        mutual_info_scores[label][feat] = 0.0
        continue
      
      N = float(total)
      N11 = float(joint_ctr[label][feat])
      N10 = float(sum([joint_ctr[l][feat] for l in label_values]) - N11)
      N01 = float(label_ctr[label] - N11)
      N00 = float(total - N11 - N10 - N01)

      N1x = N10 + N11;
      N0x = N00 + N01;
      Nx1 = N01 + N11;
      Nx0 = N00 + N10;

      #if a feature only ever occurs in a class, mi = 1
      if N10 == 0 or N01 == 0: 
        mutual_info_scores[label][feat] = 10.0
        continue

      if chi:
        """ Chi2 """
        num = ((N11 + N10 + N01 + N00) * (N11 * N00 - N10 * N01) * (N11 * N00 - N10 * N01))
        denom = ((N11 + N01) * (N11 + N10) * (N10 + N00) * (N01 + N00));
        score = num/denom
      else:
        """ MI """
        num1 = (N11 / N) * math.log((N * N11) / (N1x * Nx1)) 
        num2 = (N01 / N) * math.log((N * N01) / (N0x * Nx1))
        num3 = (N10 / N) * math.log((N * N10) / (N1x * Nx0))
        num4 = (N00 / N) * math.log((N * N00) / (N0x * Nx0));
        score = num1 + num2 + num3 + num4

      mutual_info_scores[label][feat] = score
    max_score = max([mutual_info_scores[label][feat] for label in label_values])
    """ If it's not a significant feature for any of the classes, dump it """
    if max_score >= threshold:
      good_features.add(feat)

  feature_map = dict(zip(good_features, range(len(good_features))))
  features = []
  for i, tweetinfo in enumerate(training_data):
    label = tweetinfo["Answer"]
    labels.append(label_map[label])
    feature_arr = [0]*len(good_features)
    for feat in tweetinfo["Features"]:
      """ If a score is too low, don't include it """
      if feat in feature_map:
        feature_arr[feature_map[feat]] = 1.0
    features.append(feature_arr)

  npfeatures = np.array(features, dtype=np.uint8)
  nplabels = np.array(labels, dtype=np.uint8)
  return (npfeatures, mutual_info_scores, feature_map, nplabels, label_map)


def chi(data):
  return run_fs(data, 0.0001, chi=True)


def label_features(data):
  features = []
  labels = []
  for tweetinfo in data:
    featuresFound = tweetinfo["Features"]
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
  npfeatures = np.array(features, dtype=np.uint8)
  nplabels = np.array(labels, dtype=np.uint8)
  return npfeatures, nplabels
