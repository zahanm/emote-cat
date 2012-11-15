import nltk
romney_file = '../Tweet-Data/Romney-Labeled.csv'
tunisia_file = '../Tweet-Data/Tunisia-Labeled.csv'


f = open(romney_file)
for line in f.readlines():
    line = line.rstrip()
    
