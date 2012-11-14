package main.scala

import org.apache.mahout.common.RandomUtils
import io.Source
import collection.mutable.ListBuffer
import collection.immutable
import breeze.text.segment.JavaSentenceSegmenter
import breeze.text.tokenize.PTBTokenizer
import breeze.text.analyze.PorterStemmer
import breeze.linalg.{Counter, Counter2}
import java.util.regex.Pattern

/**
 * Baselineing routines
 *
 * @author gibbons4
 */
object Baseline {

  protected final val STOP_LIST = immutable.Set("rt", "a", "the", "...")
  protected final val CSV_PATTERN = Pattern.compile("""(?:(?<=")([^"]*)(?="))|(?<=,|^)([^,]*)(?=,|$)""")
  protected final val URL_PATTERN =
    Pattern.compile("""\b(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]""")
  protected final val URL_TOKEN = "<URL>"
  protected var vocabulary = Counter[String, Int]()


  /**
   * Doesn't work and I'm giving up
   * @param source
   */
  def readCSV (source: String) = {
    val lines = Source.fromFile(source).getLines()
    lines.toList map(line => {
      val fields = line.split(",")
      //println(fields(1), fields(3), fields(5))
      (fields(1), fields(3), fields(5))
    })
  }

  /**
   * Yay fakeData
   * @return
   */
  def fakeData : List[immutable.Map[String, String]] = {
    val tweetData = ListBuffer.empty[immutable.Map[String, String]]
    tweetData += immutable.Map("tweet" -> "My Main fear in all that is going in #tunisia is the fate of the animal farm by G O.get rid of one thief to replace him with 10 more", "category" -> "afraid")
    tweetData += immutable.Map("tweet" -> "Theres no coverage on tv about the flood in brazil or the riots in tunisia? Yet they showing the shootings that happened in arizona? #WTF", "category" -> "angry")
    tweetData += immutable.Map("tweet" -> "New Tunisia Update: A: Australian students trapped in Tunisia among the vi... http://liveword.ca/go/117 #sidibouzid #jasminrevolt #optunisia", "category" -> "anxious")
    return tweetData.toList
  }

  /**
   * Pull out what we're going to be looking for
   * @param data
   */
  def extractTextFeatures (data : Iterable[String]) : Iterable[Counter[String, Int]] = {
    val featureVectors = ListBuffer.empty[Counter[String, Int]]
    for (tweet <- data) {
      val tokenized = PTBTokenizer(tweet.toLowerCase)
      val stemmed = tokenized.map( word => (new PorterStemmer)(word) )
      val pruned = transforms(stemmed)
      val ngrams = Counter[String, Int]()
      for (word <- pruned) {
        vocabulary(word) += 1
        ngrams(word) += 1
      }
      featureVectors += ngrams
    }
    return featureVectors.toList
  }

  /**
   * Text transforms
   *
   * - stop word filtering
   * - normalizing urls to "<URL>"
   *
   * @param tokens
   * @return
   */
  def transforms (tokens : Iterable[String]) : Iterable[String] = {
    tokens
      .filter( tok => !(STOP_LIST contains tok) )
      .map(tok => if (URL_PATTERN.matcher(tok).matches()) URL_TOKEN else tok)
  }

  def trainDevTest(input : List[Map[String, String]]) = {
    val totalSize = input.size
    val train = (input.take((totalSize * .8).toInt)).toSet
    val dev = (input.take((totalSize * .1).toInt)).toSet
    val test = input.filter(x => !(train.contains(x) || dev.contains(x)))
    (train.toList, dev.toList, test)
  }

  def bagger(datum: Map[String, String]) : Map[String, String] = {
    //{datum.get("tweet").get.split(" ").flatMap(word => List((word, datum.get("category").get), (datum.get("category2").get)))}.toMap
    val words = datum.get("tweet").get.split(" ")
    val map1 = words.map(word => (word, datum.get("category").get)) toMap
    val map2 = words.map(word => (word, datum.get("category_2").get)) toMap
    val map = map1 ++ map2.map{ case (k,v) => k -> (v + map1.getOrElse(k,0)) }
    return map
  }

  // Bag of Words
  def train(trainData : List[Map[String, String]]) = {
    val wordMap = Counter2[String, String, Int]()
    println(trainData)
    println("Callling bagger")
    trainData.flatMap(datum => bagger(datum)).toList.foreach(x => wordMap(x._1, x._2) += 1)
    wordMap
  }

  /**
   * Main method
   * @param args
   */
  def main(args: Array[String]) = {
    val tunisiaData = readCSV("/afs/cs.stanford.edu/u/gibbons4/cs224n/emote-cat/Tweet-Data/Tunisia-Labeled.csv")
    //val sources = immutable.Seq("Tweet-Data/Tunisia-Labeled.csv")
    val data = tunisiaData.drop(1) map (x => immutable.Map("tweet" -> x._1, "category" -> x._2, "category_2" -> x._3))
    val dataSets = trainDevTest(data)
    val (trainData, devData, testData) = (dataSets._1, dataSets._2, dataSets._3)
    //println(train(trainData))
    train(trainData)
    println("wtf")
    /*
    val categories = trainData.map(point => point("category"))
    val featureVectors = extractTextFeatures( trainData.map(point => point("tweet")) )
    println("=================")
    println(featureVectors)
    println("=================")
    println(categories)
    println("=================")
    //kmeans(categories, featureVectors)
    */
  }

}
