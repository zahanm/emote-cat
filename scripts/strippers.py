import sys

fd = open(sys.argv[1])

x = 0;
prev = ""
for line in fd :
  line = line.rstrip('\n').rstrip('\r').split('\t')
  line = line[2:]
  '\t'.join(line)
  if line == prev : continue
  prev = line
  print line
  x += 1
  if x > 20000 : break
