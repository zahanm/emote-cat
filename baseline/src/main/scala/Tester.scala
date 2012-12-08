package main.scala
 
import org.apache.mahout.common.RandomUtils
import io.Source
import collection.mutable.ListBuffer
import collection.immutable
import collection.mutable
import breeze.text.segment.JavaSentenceSegmenter
import breeze.text.tokenize.PTBTokenizer
import breeze.text.analyze.PorterStemmer
import breeze.linalg.{Counter, Counter2}
import java.util.regex.Pattern

/**
 * Testering routines
 *
 * @author gibbons4
 */
object Tester {

  protected final val TRAIN_SIZE = 0.8
  protected final val DEV_SIZE = 0.15

  //protected final val CSV_PATTERN = Pattern.compile("""(?:(?<=")([^"]*)(?="))|(?<=,|^)([^,]*)(?=,|$)""")
  //protected final val URL_PATTERN =
  //  Pattern.compile("""\b(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]""")
  //protected final val URL_TOKEN = "<URL>"
  //protected var vocabulary = Counter[String, Int]()


  /**
  * Hardcoded to Amazon Mechanical Turk data format
   * @param source
   */
  def readCSV (source: String) = {
    val lines = Source.fromFile(source).getLines()
    lines.toList map(line => {
      val cleaner = line.replaceAllLiterally("\"\"", "\"").replaceAllLiterally("\"", "")
      val fields = cleaner.split(",")
      //println(fields(1), fields(3), fields(5))
      (fields(1), fields(3), fields(5))
    })
  }
  
  /**
  * Splits the input file into a training, development, and test file.
  */
  def trainDevTest(input : List[Map[String, String]]) = {
    val totalSize = input.size
    val train = (input.take((totalSize * TRAIN_SIZE).toInt)).toSet
    val dev = (input.take((totalSize * DEV_SIZE).toInt)).toSet
    val test = input.filter(x => !(train.contains(x) || dev.contains(x)))
    println("Total Size:\t" + input.size)
    println("Train Size:\t" + train.size)
    println("Dev   Size:\t" + dev.size)
    println("Test   Size:\t" + test.size)
    (train.toList, dev.toList, test)
  }

  def kFold(input : List[Map[String, String]], k : Int = 10) = {
    println("Total Size:\t" + input.size)
    println("Using " + k + "-fold validation")
    val segments = input.grouped((input.size + k-1) / k).toList
    segments map {s => (segments filter (x => x != s) reduce (_ ++ _), s)}
  }

  def printInfo(data : List[Map[String, String]]) {
    val counts = (data flatMap(x => x filter(kv => kv._1.contains("cat")))).map(kv => kv._2)
    .groupBy(identity).map(kv =>(kv._1, kv._2.size))
    counts.foreach(x => println("\t" + x._1 + "\t" + x._2))
    println("\ttotal\t" + counts.map(x => x._2).reduce(_+_))
  }

  def printStats(trainData : List[Map[String, String]], devData : List[Map[String, String]]) {
    println("=== training information === ")
    printInfo(trainData)
    println("=== evaluation information === ")
    printInfo(devData)
    println("====================================")
  }
    //for recall...
    var label_true_totals = collection.mutable.HashMap[String, Int]()
    var label_guess_totals = collection.mutable.HashMap[String, Int]()
    var label_correct = collection.mutable.HashMap[String, Int]()
  def runData(classifier : Classifier, trainData : List[Map[String, String]], evalData: List[Map[String, String]]) = {
    //label_true_totals = collection.mutable.HashMap[String, Int]()
    //label_correct = collection.mutable.HashMap[String, Int]()
    classifier.train(trainData)
    val guesses = evalData map (datum => {
      val guess : String = classifier.classify(datum.get("tweet").get)
      val correct = if (guess == datum.get("category").get || guess == datum.get("category_2").get) 1 else 0
        label_true_totals.put(datum.get("category").get, label_true_totals.getOrElse(datum.get("category").get, 0) + 1)
        label_true_totals.put(datum.get("category_2").get, label_true_totals.getOrElse(datum.get("category_2").get, 0) + 1)
        label_guess_totals.put(guess, label_guess_totals.getOrElse(guess, 0) + 1)
      if(correct == 1) {
        label_correct.put(guess, label_correct.getOrElse(guess, 0) + 1)
      }
      val explore = "sad" 
      if(correct == 0 && (datum.get("category").get == explore || datum.get("category_2").get == explore)) {
       //println(guess + "\t: " + datum.get("tweet").get)
      }
      (guess, correct)
    })
    guesses
  }
  
  /**
   * Main method
   * @param args
   */
  def main(args: Array[String]) = {
    val helpString = "run-main main.scala.Tester [ -f filename -c Categorizer -e dev -n NoneTagBoolean]"
    val pp = new ParseParms(helpString)
    pp.parm("-f", "Tweet-Data/Tunisia-Labeled.csv")
      .parm("-c", "main.scala.Baseline")
      .parm("-e", "dev")
      .parm("-n", "true") // i'll fix it later...
    val parsed_args = pp.validate(args.toList)._3
    val labeledData = readCSV(parsed_args.get("-f").get)
    var data = labeledData.drop(1) map (x => immutable.Map("tweet" -> x._1, "category" -> x._2, "category_2" -> x._3))
    if(parsed_args.get("-n").get == "true") data = data filter (x => x.get("category").get != "none" || x.get("category_2").get != "none")
    val classifier : Classifier = Class.forName(parsed_args.get("-c").get).newInstance.asInstanceOf[Classifier]
    val k_data = kFold(data)
    //val label_correct = collection.mutable.HashMap[String, Int]()
    val totals = k_data map {data =>
      val guesses = runData(classifier, data._1, data._2)
      val scores = guesses groupBy(_._1) map (scores => {
        val good = scores._2.map(x=>x._2).reduce(_+_) 
        val score = good / scores._2.size.toFloat
        //label_correct.put(scores._1, label_correct.getOrElse(scores._1, 0) + good)
        (scores._1, good, scores._2.size)
      }) 
      val total = scores.reduce((acc, n) => ("total", (acc._2 + n._2), (acc._3 + n._3)))
      total
    } 
  
    println("--------------------------------")
    label_correct foreach {case(label, value) => println(label + " : " + value + " / " + label_true_totals.get(label).get + " / " + label_guess_totals.get(label).get + " = " + (value / label_true_totals.get(label).get.toFloat) + " : " + (value / label_guess_totals.get(label).get.toFloat))}
    val total = totals reduce { (acc, n) => ("total", acc._2 + n._2, acc._3 + n._3)}
    println("--------------------------------")
    println("total\t:\t" + total._2 + " / " + total._3 + " = " + (total._2.toFloat / total._3))
  }
}
