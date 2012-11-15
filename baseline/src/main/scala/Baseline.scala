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
class Baseline extends Classifier {

  protected var wordMap : Map[String, Map[String, Int]] = null

  protected final val TRAIN_SIZE = 0.8
  protected final val DEV_SIZE = 0.1

  protected final val STOP_LIST = immutable.Set("rt", "a", "the", "...")
  protected final val CSV_PATTERN = Pattern.compile("""(?:(?<=")([^"]*)(?="))|(?<=,|^)([^,]*)(?=,|$)""")
  protected final val URL_PATTERN =
    Pattern.compile("""\b(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]""")
  protected final val URL_TOKEN = "<URL>"
  protected var vocabulary = Counter[String, Int]()

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

  def bagger(datum: Map[String, String]) : List[(String, String)] = {
    //{datum.get("tweet").get.split(" ").flatMap(word => List((word, datum.get("category").get), (datum.get("category2").get)))}.toMap
    val words = transforms(datum.get("tweet").get.split(" "))
    val count1 = words.map(word => (word, datum.get("category").get))
    val count2 = words.map(word => (word, datum.get("category_2").get))
    //val map = map1 ++ map2.map{ case (k,v) => k -> (v + map1.getOrElse(k,0)) }
    (count1 ++ count2).toList
  }

  // Bag of Words
  def train(trainData : List[Map[String, String]]) {
    //val wordMap = Counter2[String, String, Int]()
    //println(trainData)
    //println("Callling bagger")
    //trainData.flatMap(datum => bagger(datum)).toList.foreach(x => wordMap(x._1, x._2) += 1)
    wordMap = trainData.flatMap(datum => bagger(datum))
    .groupBy(_._1)
    .map(kv =>(kv._1, kv._2 map (x => x._2)))
    .map( kv => (kv._1, kv._2 groupBy identity mapValues(_.size)))
  }

  def classify(tweet: String) = {
    val emotionCounts = Counter[String, Int]()
    tweet.split(" ") foreach (word => {
      if(wordMap.contains(word)) {
        wordMap.get(word).get foreach(kv => emotionCounts(kv._1) += kv._2)
      }
    })
    emotionCounts.argmax(Ordering[Int])
  }
}
