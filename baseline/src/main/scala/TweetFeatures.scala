package main.scala

import scala.util.matching.Regex
import scala.math._

import org.apache.mahout.common.RandomUtils
import io.Source
import scala.io.Source._
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
class TweetFeatures extends Classifier {

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

  // Determine if there is at least one word that is all capitalized
  def isShouting(tweet: String) : Boolean = {
    val tokenized = Twokenize(tweet)
    val uppers = Twokenize(tweet.toUpperCase).toSet
    tokenized foreach {word => 
      if(word.length > 1 && uppers.contains(word) && word != "RT" && word.charAt(0).isLetter) {
        //println("RAAAWWWWRRRRR!  : " + word)
        return true
      }
    }
    return false
  }

  //Looks for RT or @
  def dialogue(tweet: String) : Boolean = {
    val tokenized = Twokenize(tweet)
    tokenized foreach { word =>
      if(word.contains("RT") || word.contains("@")) 
        return true
    }
    return false
  }

  //checks for membership in a hardcoded list of negation words
  val negativeWords = Set("no", "not", "never", "cannot", "wont", "cant", "dont", "n't")
  def isNegated(tweet: String) : Boolean = {
    val tokenized = Twokenize(tweet.toLowerCase)
    tokenized foreach { word => 
      if(negativeWords.contains(word))
        return true
    } 
    return false
  }

  //checks for emoticons and only separates happy mouths from sad mouths from other
  val emojiRegex = new Regex( Twokenize.emoticon)
  def emoticonTags(tweet: String) : String = {
    val emoji = emojiRegex.findAllIn(tweet.replaceAll("://", "")).toList
    if(emoji.length == 0) return "" //return " $-no-face"
    val last = emoji.last
    if(Twokenize.sadMouths.indexOf(last) > -1) {
     // println("sad: " + last)
      return " $sad-face"
    } else if(Twokenize.happyMouths.indexOf(last) > -1) {
     // println("happy: " + last)
      return " $happy-face"
    } else {
     // println("other: " + last)
      return " $other-face"
    }
  }

  //checks for dramatic punctuation
  def dramaticPunctuation(tweet: String) : String = {
    if(tweet.indexOf("?!") > -1 || tweet.indexOf("!?") > -1 || tweet.indexOf("??") > -1) {
      return " $-dramatic-?"
    } else if(tweet.indexOf("!!") > -1) {
      return "$-dramatic-!"
    }
    //return "$-boring"
    return ""
  }

  //checks to see if its trending #hash-tag
  def hasHashTag(tweet: String) : Boolean = {
    return tweet.indexOf('#') > -1
  }

  //Incorporate Senti-net for a positive negative score
  val SWN = scala.io.Source.fromFile("Tweet-Data/SWN_tags.txt", "utf-8").getLines map {line =>
    val s = line.toLowerCase.split(' ')
    (s(0), s(1).toInt)
  } toMap

  def sentimentWords(tweet: String) : String = {
    val tokenized = Twokenize(tweet.toLowerCase)
    val pos = {tokenized filter (x => SWN.getOrElse(x, 2) == 1)}.size
    val neg = {tokenized filter (x => SWN.getOrElse(x, 2) == 0)}.size
    if ((pos == 0 && neg == 0) || abs(pos - neg) < 4) return "" //return " $-no-sentiment"; 
    if(pos > neg) return " $-pos-sent"
    return " $-neg-sent"
  }

  /**
  *  Applies stemming and pruning to the words in the tweet
  *
  *  @param tweet
  *  @return clean tweet
  */
  //var count = 5
  def cleaner(raw_tweet : String) = {
    val tweet = {
      var t = raw_tweet
      if(isShouting(raw_tweet)) t += " $shout"
      t += emoticonTags(raw_tweet) //ie " $happy"
      if(isNegated(raw_tweet)) t += " $negation"
      if(dialogue(raw_tweet)) t += " $dialogue"
      t += dramaticPunctuation(raw_tweet)
      //if(hasHashTag(raw_tweet)) t += " $-#"
      t += sentimentWords(raw_tweet)
      t
    }
    val tokenized = PTBTokenizer(tweet.toLowerCase)
    val stemmed = tokenized.map( word => (new PorterStemmer)(word) )
    val pruned = transforms(stemmed) toList
    val doubles = for(i <- 0 until pruned.length-1) yield pruned(i) + pruned(i+1)
    val triples = for(i <- 0 until pruned.length-2) yield pruned(i) + pruned(i+1) + pruned(i+2)
    //if (count < 0) System.exit(0)
    //count -= 1
    //println("Tweet: " + tweet + " / " + pruned)
    pruned ++ doubles ++ triples
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
