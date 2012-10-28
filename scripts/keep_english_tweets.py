import sys

if(len(sys.argv)) != 2 :
  print 'usage: python2.6 keep_english_tweets.py tweetFile wordFile'

dict = open(sys.argv[2]).read().splitlines()
print "Using " + str(len(dict)) + " words"

tweets = open(sys.argv[1])
for line in tweets : 
  words = line.lower().rstrip().split(' ')
  good = 0
  for word in words :
    if word in dict : 
      #print word
      good += 1
  #print good,
  #print len(words)
  if good > (len(words)/2) : print line,
