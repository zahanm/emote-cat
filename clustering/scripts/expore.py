
from glob import iglob as glob
import DictReader from csv

def iterdata(source):
  with open(source) as f:
    reader = DictReader(f)
    for line in reader:
      yield line

def explore():
  trainfile = "../../Tweet-Data/Romney-Labeled.csv"
  for data in iterdata(trainfile):
    data["Tweet"]
    data["Answer1"]
    data["Answer2"]
    data["Agreement"]

if __name__ == "__main__":
  explore()
