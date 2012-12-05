import sys

#pass in a SWN file
fd = open(sys.argv[1])

for line in fd :
  line = line.split('\t')
  if len(line) < 5 : continue
  pos = float(line[2])
  neg = float(line[3])
  words = line[4].split(' ')
  for w in words :
    w = w[:w.find('#')]
    if pos - neg >= 0.25 :
      print w, 1
    if neg - pos >= 0.25 :
      print w, 0

