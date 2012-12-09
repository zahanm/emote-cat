
from datetime import datetime, timedelta
import argparse
import re
import itertools
import os
import os.path as path
import gzip

import numpy as np
import matplotlib.pyplot as plt

def conv(inp):
  with open(inp) as f:
    freqs = []
    for line in f:
      numbers = re.split(r",", line)
      m = re.search(r"\d+", numbers[0])
      times.append(m.group())
      m = re.search(r"\d+", numbers[1])
      freqs.append(m.group())
  starting = datetime(2012, 11, 6, 15, 0, 0)
  with open('output.txt', 'w') as out:
    for t, f in zip(times, freqs):
      now = starting + timedelta(minutes= t * 10)
      out.write("(")
      out.write(now.strftime("%I:%M"))
      out.write(", ")
      out.write(f)
      out.write(")")
      out.write("\n")

def plottimes(inp):
  parts = path.basename(inp).split(".")[0].split("_")
  model, emotion = parts[:2]
  data = "_".join(parts[2:])
  starting = datetime(2012, 11, 6, 11, 50, 0)
  ending = datetime(2012, 11, 7, 0, 30, 0)
  times = []
  buckets = []
  totals = []
  i = 0
  while (starting + timedelta(minutes= i * 50)) <= ending:
    times.append(starting + timedelta(minutes= i * 50))
    buckets.append(0)
    totals.append(0)
    i += 1
  obama_freqs = np.zeros((len(buckets),), dtype=int)
  romney_freqs = np.zeros((len(buckets),), dtype=int)
  splitter_pat = re.compile(r"\t")
  obama_pat = re.compile(r"barack|obama", re.I)
  romney_pat = re.compile(r"mitt|romney", re.I)
  with gzip.open(inp) as f:
    # header
    f.next()
    gg = 0
    for line in f:
      items = splitter_pat.split(line)
      dt = datetime.strptime(items[0], "%Y-%m-%d %H:%M:%S")
      tweet = items[1]
      label = int(items[2])
      while dt > times[gg]:
        gg += 1
      totals[gg] += 1
      if label == 1:
        buckets[gg] += 1
        if obama_pat.search(tweet):
          obama_freqs[gg] += 1
        elif romney_pat.search(tweet):
          romney_freqs[gg] += 1
  freqs = np.array(buckets, dtype=float)
  norms = np.array(totals, dtype=float)
  # remove zeros
  freqs[norms == 0] = 0.0
  obama_freqs[norms == 0] = 0.0
  romney_freqs[norms == 0] = 0.0
  norms[norms == 0] = 1.0
  # skip 0, 6, 12 th ones
  times = times[1:6] + times[7:12] + times[13:]
  freqs = np.concatenate((freqs[1:6], freqs[7:12], freqs[13:]))
  norms = np.concatenate((norms[1:6], norms[7:12], norms[13:]))
  obama_freqs = np.concatenate((obama_freqs[1:6], obama_freqs[7:12], obama_freqs[13:]))
  romney_freqs = np.concatenate((romney_freqs[1:6], romney_freqs[7:12], romney_freqs[13:]))
  for t, b in itertools.izip(times, buckets):
    print "({}, {})".format(t.strftime("%H:%M"), b)
  # time shift
  timevalues = map(lambda t: int(t.strftime("%s")) - int(starting.strftime("%s")), times)
  timenames = map(lambda t: (t + timedelta(hours=3)).strftime("%I:%M %p"), times)
  # space for labels
  plt.gcf().subplots_adjust(bottom=0.15)
  # mkdir "plots"
  if not path.exists("plots"):
    os.mkdir("plots")
  if not path.exists(path.join("plots", emotion)):
    os.mkdir(path.join("plots", emotion))
  # overall raw vs time
  # ---
  mkplot("raw", emotion, data, timevalues, timenames, [ (freqs, "Overall") ], "Number of tweets")
  # romney and obama raw vs time
  # ---
  mkplot("partisan", emotion, data, timevalues, timenames, [ (romney_freqs / norms, "Romney"), (obama_freqs / norms, "Obama"), ], "Percentage")
  # percentage vs time
  # ---
  mkplot("percent", emotion, data, times, timevalues, [ (freqs / norms, "Overall") ], "Percentage")

def mkplot(pltname, emotion, data, xs, xnames, ys, ylabel):
  for y, ytype in ys:
    plt.plot(xs, y, ".-", label=ytype)
  plt.xticks(xs, xnames, rotation=45)
  plt.ylabel(ylabel)
  if len(ys) > 1:
    plt.legend()
  plt.title(emotion[:1].upper() + emotion[1:] + " vs Time: " + data)
  out_fname = "{}_{}.png".format(pltname, data)
  print "Writing to: {}".format(path.join("plots", emotion, out_fname))
  plt.savefig(path.join("plots", emotion, out_fname))
  plt.clf()

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(title='Sub commands')

parser_convert = subparsers.add_parser('convert', help='Convert times to LaTeX')
parser_convert.add_argument("data")
parser_convert.set_defaults(func=conv)

parser_graph = subparsers.add_parser('graph', help='Plot output from "emote.py predict"')
parser_graph.add_argument("data", help="data, gzipped")
parser_graph.set_defaults(func=plottimes)

ARGV = parser.parse_args()

if __name__ == '__main__':
  ARGV.func(ARGV.data)
