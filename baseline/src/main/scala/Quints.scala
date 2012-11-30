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
 * Doubleing routines
 *
 * @author gibbons4
 */
class Quints extends Classifier {

  protected var wordMap : Map[String, Map[String, Float]] = null

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
  *  Applies stemming and pruning to the words in the tweet
  *
  *  @param tweet
  *  @return clean tweet
  */
  def cleaner(tweet : String) = {
    val tokenized = PTBTokenizer(tweet.toLowerCase)
    val stemmed = tokenized.map( word => (new PorterStemmer)(word) )
    val pruned = transforms(stemmed) toList
    val doubles = for(i <- 0 until pruned.length-1) yield pruned(i) + pruned(i+1)
    val triples = for(i <- 0 until pruned.length-2) yield pruned(i) + pruned(i+1) + pruned(i+2)
    val quads = for (i <- 0 until pruned.length-3) yield pruned(i) + pruned(i+1) + pruned(i+2) + pruned(i+3)
    val quints = for (i <- 0 until pruned.length-4) yield pruned(i) + pruned(i+1) + pruned(i+2) + pruned(i+3) + pruned(i+4)
    pruned ++ doubles ++ triples ++ quads ++ quints
  }

  /**
  * @param some input string
  * @return if the string had characters, those characters without punctuation. otherwise, the string.
  */
  def stripPunctuation(input :String) : String = {
    if(("\\w".r findFirstIn input) == None) return input;
    else return input.replaceAll("[^a-z\\sA-Z]",""); 
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
      .map(tok => stripPunctuation(tok))
  }

  /**
  * Count the (word, label) occurences
  *
  * @param a list of tweets (represented as  map, indexable with tweet, category, category_2)
  * @return a list of (word, label)
  */
  def bagger(datum: Map[String, String]) : List[(String, String)] = {
    val words = cleaner(datum.get("tweet").get)
    val count1 = words.map(word => (word, datum.get("category").get))
    val count2 = words.map(word => (word, datum.get("category_2").get))
    (count1 ++ count2).toList
  }

  def train(trainData : List[Map[String, String]]) {
    wordMap = trainData.flatMap(datum => bagger(datum))
    .groupBy(_._1)
    .map(kv =>(kv._1, kv._2 map (x => x._2)))
    .map(kv => (kv._1, kv._2 groupBy identity mapValues(_.size.toFloat)))
    val totals = wordMap map (kv => (kv._1, kv._2.values.reduce(_+_)))
    wordMap = wordMap map (kv => (kv._1, kv._2 map (xy => (xy._1, xy._2.toFloat / totals.get(kv._1).get))))
    wordMap
  }

  def classify(tweet: String) : String = {
    val emotionCounts = Counter[String, Float]()
    cleaner(tweet) foreach (word => {
      if(wordMap.contains(word)) {
        wordMap.get(word).get foreach(kv => emotionCounts(kv._1) += kv._2)
      }
    })
    if(emotionCounts.size == 0) {
      println("Needs help: " + tweet)
      return "no labels found"
    }
    emotionCounts.argmax(Ordering[Float])
  }
}
