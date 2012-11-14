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
 * Hello world simple.
 * @author zahanm
 */
object Cluster {

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
    var header = ListBuffer
    var matcher = CSV_PATTERN.matcher(lines.next)
    while (matcher.find) {
//      header += matcher.group
    }
    while (lines.hasNext) {
      val line = lines.next
      val matcher = CSV_PATTERN.matcher (line)
      while (matcher.find) {
        val col1 = matcher.group (0)
        val col2 = matcher.group (1)
        println("1: " + col1)
        println("2: " + col2)
      }
    }
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

  /**
   * Perform the kmeans computation
   *
   * @param categories
   * @param featureVectors
   */
  def kmeans(categories: Iterable[String], featureVectors: Iterable[ Counter[String, Int] ]) = {
    for (vec <- featureVectors) {
      println(vec)
    }
  }

  /**
   * Main method
   * @param args
   */
  def main(args: Array[String]) = {
    val sources = Seq("Tweet-Data/Tunisia-Labeled.csv")
    val data = fakeData
    val categories = data.map(point => point("category"))
    val featureVectors = extractTextFeatures( data.map(point => point("tweet")) )
    kmeans(categories, featureVectors)
  }

}
