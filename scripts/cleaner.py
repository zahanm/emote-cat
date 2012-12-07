
import sys
import os.path as path
import gzip
import re
import itertools
from datetime import datetime
import random

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
      length = len(header)
      numskipped = 0
      numtotal = 0
      # other lines
      for i, line in enumerate(inp):
        numtotal += 1
        items = re.split(r"\t", line.strip())
        if len(items) != length:
          numskipped += 1
          continue
          """
          screw this
          """
          if len(items) <= 1:
            continue
          elif len(items) == 2 * length:
            out.write("\t".join(items[ :length / 2]) + "\n")
            out.write("\t".join(items[length / 2: ]) + "\n")
          else:
            expected = [ r"^\w+$", r"^2012-11-06 \d{2}:\d{2}:\d{2}$", r"^\d+$", r"^[\w\-\s@]+$"]
            expected_date = "2012-11-06 "
            progress = 0
            j = 0
            unfixable = False
            while j < len(items) and not unfixable:
              item = items[j]
              if progress >= len(header):
                unfixable = True
              elif not re.match(expected[progress], item):
                if progress == 0:
                  print "line number: {}".format(i+2)
                  print "items: {}".format(items)
                  name = raw_input("name: ")
                  items.insert(j, name.strip())
                elif progress == 1:
                  print "line number: {}".format(i+2)
                  print "items: {}".format(items)
                  time = raw_input("date {} time: ".format(expected_date))
                  items.insert(j, time.strip())
                elif progress == 2:
                  items.insert(j, str(random.randint(200000000000000000, 300000000000000000)))
                else:
                  unfixable = True
                j += 1
              progress += 1
              j += 1
            if not unfixable and len(items) == len(header):
              print "writing on " + str(i+2) + " => " + "\t".join(items)
              out.write("\t".join(items) + "\n")
              continue
          print "line number: {}".format(i+2)
          print "items: {}".format(items)
          skip = raw_input("skip? (y/n): ")
          if re.match(r"y(es)?", skip, re.I):
            continue
          else:
            print "No skip"
            sys.exit(2)
        expected_names = ["name", "date", "id", "tweet"]
        expected = [ r"^\w+$", r"^2012-11-06 \d{2}:\d{2}:\d{2}$", r"^\d+$", r".+"]
        for name, pat, item in itertools.izip(expected_names, expected, items):
          if not re.match(pat, item):
            print "line number: {}".format(i+2)
            print "items: {}".format(items)
            print "not matching {} : {}".format(name, item)
            skip = raw_input("exit? (y/n): ")
            if re.match(r"y(es)?", skip, re.I):
              sys.exit(2)
          """
          all done
          """
        out.write(line)
  print "Skipped {} out of {}".format(numskipped, numtotal)

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print "usage: {} <uncleaned fname>"
    sys.exit(1)
  clean( sys.argv[1] )
