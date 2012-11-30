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

  def bagger(datum: Map[String, String]) : List[(String, String)] = {
    val words = datum.get("tweet").get.split(" ")
    val count1 = words.map(word => (word, datum.get("category").get))
    val count2 = words.map(word => (word, datum.get("category_2").get))
    (count1 ++ count2).toList
  }

  // Bag of Words
  def train(trainData : List[Map[String, String]]) {
    wordMap = trainData.flatMap(datum => bagger(datum))
    .groupBy(_._1)
    .map(kv =>(kv._1, kv._2 map (x => x._2)))
    .map( kv => (kv._1, kv._2 groupBy identity mapValues(_.size)))
  }

  def classify(tweet: String) : String = {
    val emotionCounts = Counter[String, Int]()
    tweet.split(" ") foreach (word => {
      if(wordMap.contains(word)) {
        wordMap.get(word).get foreach(kv => emotionCounts(kv._1) += kv._2)
      }
    })
    if (emotionCounts.size > 0)
      return emotionCounts.argmax(Ordering[Int])
    else
      return "gobble gobble"
  }
}
