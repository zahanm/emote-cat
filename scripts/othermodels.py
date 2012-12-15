
import functools
from itertools import imap

import numpy as np

class NaiveBayes(dict):

  def train(self, features, labels):
    self["types"] = frozenset(labels)
    overalltotal = np.sum(features)
    for label in self["types"]:
      self[label] = {}
      counts = np.sum(features[labels == label, :], 0)
      total = float(np.sum(counts))
      if total == 0:
        self[label]["probs"] = counts
      else:
        self[label]["probs"] = counts / total
      self[label]["prior"] = total / overalltotal
    return self

  def apply(self, datum):
    def prob(datum, label):
      lp = self[label]["prior"]
      lp *= np.dot(datum, self[label]["probs"])
      return lp, label
    maxp = -float("inf")
    maxlabel = None
    for p, label in imap(functools.partial(prob, datum), self["types"]):
      if p > maxp:
        maxp = p
        maxlabel = label
    return maxlabel
