import numpy as np
from collections import Counter
import nltk
import twokenize
import functools
import re
import itertools
import emoticons
porter = nltk.PorterStemmer()

stoplist = frozenset(["mitt", "romney", "barack", "obama", "the", "a", "is\
", "rt", "barackobama"])

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

  """ First, see which features are significant """
  feature_threshold = 100
  feature_counter = Counter()
  for tweetinfo in training_data:
    for feat in tweet_features(tweetinfo):
      feature_counter[feat] += 1

  # produce featureMap and extract features together
  for tweetinfo in training_data:
    # add features to tweetvector
    tweetvector = [0] * numfeatures
    for feat in tweet_features(tweetinfo):
      if feature_counter[feat] < feature_threshold: continue
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


def mutualinfo(training_data):
  """ We need p(word & label) p(word) p(label) """
  print "using mutual information"
  """ First, see which features are significant """
  feature_counter = Counter()
  label_counter = Counter()
  label_feature_counter = defaultdict(Counter)
  total_tweets = 0
  for tweetinfo in training_data:
    total_tweets += 1
    #Count features at most once
    for feat in set(tweet_features(tweetinfo)):
      feature_counter[feat] += 1
      label_counter[tweetinfo["Answer"]] += 1
      label_feature_counter[label][feat] += 1

  """ Now calculate all of the mutual information scores """
  #calculate scores for all words in all labels
  mutual_info_scores = defaultdict(dict)
  for tweetinfo in training_data:
    #Count features at most once
    for feat in set(tweet_features(tweetinfo)):
      label = tweetinfo["Answer"]
      P_feat_label = label_feature_counter[label][feat] / total_tweets
      P_feat = feature_counter[feat] / sum(feature_counter.values())
      P_label = label_counter[label] / sum(label_counter.values())
      score = P_feat_label * math.log((P_feat_label)/(P_feat*P_label))
      mutual_info_scores[label][feat] = score
