
import DictReader from csv

def iterdata(source):
  with open(source) as f:
    reader = DictReader(f)
    for line in reader:
      yield line

def explore():
  traindir = "../../Tweet-Data/"

if __name__ == "__main__":
  explore()
