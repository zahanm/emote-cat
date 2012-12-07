
import sys
import os.path as path
import gzip
import re
import itertools
from datetime import datetime

from datareaders import TSVReader

def clean(inp_fname):
  with open(inp_fname) as inp:
    loc, basename = path.split(inp_fname)
    out_fname = path.join(loc, "clean_" + basename + ".gz")
    print "Cleaning to: {}".format(out_fname)
    with gzip.open(out_fname, "w") as out:
      # header line
      header = inp.next().strip().split("\t")
      out.write("\t".join(header) + "\n")
      # other lines
      for i, line in enumerate(inp):
        items = re.split(r"\t", line.strip())
        if len(items) != len(header):
          print "line number: {}".format(i+2)
          print "items: {}".format(items)
          skip = raw_input("skip? (y/n): ")
          if re.match(r"y(es)?", skip, re.I):
            continue
          else:
            print "No skip"
            sys.exit(2)
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
            print "line number: {}".format(i+2)
            print "line: {}".format(line.rstrip())
            print "Bad Line"
            sys.exit(2)
            # print "WARN: Are you sure that the datafile has a header line?"
        out.write(line)
  print "Done"

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print "usage: {} <uncleaned fname>"
    sys.exit(1)
  clean( sys.argv[1] )
