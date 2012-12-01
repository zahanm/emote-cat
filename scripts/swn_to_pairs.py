import sys

#pass in a SWN file
fd = open(sys.argv[1])

for line in fd :
 line = line.split(' ')
