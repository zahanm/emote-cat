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
import org.apache.mahout.clustering.kmeans.KMeansDriver

/**
 * Clustering routines
 * Using KMeans for now
 *
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
  def readCSV (source: String) : Iterator[Map[String, String]] = {
    val lines = Source.fromFile(source).getLines()
    lines.drop(1).map { data =>
      val point = data.split(",")
      immutable.Map("id" -> point(0), "tweet" -> point(1), "cat1" -> point(3),
        "cat2" -> point(5), "consensus" -> point(7))
    }
  }

  /**
   * Pull out what we're going to be looking for
   * @param data
   */
  def extractTextFeatures (data : Iterator[String]) : Iterator[Counter[String, Int]] = {
    data.map { tweet =>
      val tokenized = PTBTokenizer(tweet.toLowerCase)
      val stemmed = tokenized.map( word => (new PorterStemmer)(word) )
      val pruned = transforms(stemmed)
      val ngrams = Counter[String, Int]()
      pruned.foreach { word =>
        vocabulary(word) += 1
        ngrams(word) += 1
      }
      ngrams
    }
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
   * Chi-sq feature selection
   *
   * @param categories
   * @param featureVectors
   */
  def chiSq(categories: Counter[String, Int], featureVectors: Iterator[ Counter[String, Int] ]) = {
    val k = categories.size

    categories.keysIterator.map { cat =>
      val pcat = categories(cat).toDouble / categories.sum
      vocabulary.keysIterator.map { word =>
        val pword = vocabulary(word).toDouble / vocabulary.sum
        val pwordgivencat = featureVectors.foldLeft(0.0) { (total, vector) => total + vector(word) } / vocabulary(word)

      } .toList
    }
  }

  /**
   * Main method
   * @param args
   */
  def main(args: Array[String]) = {
    val sources = immutable.Seq("Tweet-Data/Tunisia-Labeled.csv")
    val data = readCSV(sources(0))
    val categories = Counter[String, Int]()
    data.foreach(point => categories(point("cat1")) += 1)
    val featureVectors = extractTextFeatures( data.map(point => point("tweet")) )
    chiSq(categories, featureVectors)
  }

}
