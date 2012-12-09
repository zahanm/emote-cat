
from datetime import datetime, timedelta
import argparse
import re
import itertools
import os
import os.path as path

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
  parts = path.splitext(path.basename(inp))[0].split("_")
  model, emotion = parts[:2]
  data = "_".join(parts[2:])
  now = datetime(2012, 11, 6, 11, 50, 0)
  ending = datetime(2012, 11, 7, 0, 30, 0)
  times = []
  buckets = []
  totals = []
  i = 0
  while (now + timedelta(minutes= i * 50)) <= ending:
    times.append(now + timedelta(minutes= i * 50))
    buckets.append(0)
    totals.append(0)
    i += 1
  splitter_pat = re.compile(r"\t")
  with open(inp) as f:
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
  freqs = np.array(buckets, dtype=float)
  norms = np.array(totals, dtype=float)
  # remove zeros
  freqs[norms == 0] = 0.0
  norms[norms == 0] = 1.0
  for t, b in itertools.izip(times, buckets):
    print "({}, {})".format(t.strftime("%H:%M"), b)
  plt.plot(times, freqs / norms)
  # time shift
  timenames = map(lambda t: (t - timedelta(hours=3)).strftime("%H:%M"), times)
  plt.xticks(times, timenames, rotation=45)
  plt.ylabel("Percentage")
  plt.title(emotion[:1].upper() + emotion[1:] + " vs Time: " + data)
  out_fname = "plot_{}_{}.png".format(emotion, data)
  if not path.exists("plots"):
    os.mkdir("plots")
  print "Writing to: {}".format(path.join("plots", out_fname))
  plt.savefig(path.join("plots", out_fname))

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(title='Sub commands')

parser_convert = subparsers.add_parser('convert', help='Convert times to LaTeX')
parser_convert.add_argument("data")
parser_convert.set_defaults(func=conv)

parser_graph = subparsers.add_parser('graph', help='Plot output from "emote.py predict"')
parser_graph.add_argument("data")
parser_graph.set_defaults(func=plottimes)

ARGV = parser.parse_args()

if __name__ == '__main__':
  ARGV.func(ARGV.data)
