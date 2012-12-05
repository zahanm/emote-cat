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
import scala.math.{exp, pow, random, sqrt, log}

/**
 * Builds a neural network and uses it to classify future data points
 *
 * @author gibbons4
 */
class FastNN extends Triples {//Classifier {
  import collection.mutable.HashMap

  def sigmoid ( value: Float) : Float = {
    1f / (1f + exp(-value).toFloat)
  }

  def vectorSigmoid( v : Array[Float]) = {
    v map {x => sigmoid(x)} toArray
  }

  // input to hidden layer. v is sparse, thus hashmap
  def vectorMatrixMultiply(v : Map[Int, Float], m : Array[Array[Float]]) = {
    val b = new Array[Float](m.size)
    for (column <- 0 until m.size) {
      b(column) = v map {case(index, f_value) => f_value * m(column)(index)} reduce (_+_)
    }
    b 
  }

  def vectorMatrixMultiply(v : Array[Float], m : Array[Array[Float]]) = {
    val b = new Array[Float](m.size)
    for (column <- 0 until m.size) {
      b(column) = (for(i <- 0 until v.size) yield m(column)(i) * v(i) ) reduce(_+_)
    } 
    b
  }

  def normalize(a : Array[Float]) = {
    val ttl = a reduce (_+_)
    for (i <- 0 until a.length) {
      a(i) = a(i) / ttl
    }
    a
  }

  // TODO : what should this be
  def costFunction(a : Array[Float], b : Array[Float]) = {
    //a.zip(b) map { case(x, y) => ((y-x)*(y - x)).toFloat}//*(1-x)*(x).toFloat}
    val _error : Float = sqrt(a.zip(b) map { case(x, y) => (y-x)*(y-x)} reduce (_+_)).toFloat 
    a.map(x => _error).toArray
  }

  var totalLabels = 8 // Bleh!
  var numLabels = 0
  val labelPositions = HashMap[String, Int]()
  val labelIndex = HashMap[Int, String]()

  def trueVector(a : String, b : String) = {
    if(!labelPositions.contains(a)) {labelPositions.put(a, numLabels); labelIndex.put(numLabels, a); numLabels += 1}
    if(!labelPositions.contains(b)) {labelPositions.put(b, numLabels); labelIndex.put(numLabels, b); numLabels += 1}
    val trueLabels = new Array[Float](totalLabels) //primtive Float initializes to zero
    trueLabels(labelPositions.get(a).get) += 0.5f
    trueLabels(labelPositions.get(b).get) += 0.5f
    trueLabels
  }

  var numWords = 0
  val wordPositions = HashMap[String, Int]()

  def featureVector(features : List[String]) = {
    features map { case(word) =>
      if(!wordPositions.contains(word)) {wordPositions.put(word, numWords) ; numWords += 1}
      (wordPositions.get(word).get, 1f / features.size.toFloat)
    } toMap
  }

  def classifyVector(features : List[String]) = {
    features filter (w => wordPositions.contains(w)) map {w => (wordPositions.get(w).get, 1f / features.size.toFloat)} toMap
  }

  def vectorize(trainData : List[Map[String, String]]) = {
    trainData map { sentenceMap =>
      (featureVector(cleaner(sentenceMap.get("tweet").get)), trueVector(sentenceMap.get("category").get, sentenceMap.get("category_2").get))
    }
  }

  //TODO - what should this be
  val hiddenSize = 1000
  
  def randomInit(wordList : Iterable[String]) = {
    // Weight matrices to change dimensionality
    //index by COLUMNS
    inputWeights = new Array[Array[Float]](hiddenSize)
    outputWeights = new Array[Array[Float]](numLabels)

    val rng = new scala.util.Random(1998L)

    println("Using : " + numLabels + " and " + numWords)
  
    for(i <- 0 until inputWeights.size) {
      val inputColumn = new Array[Float](numWords)
      for(j <- 0 until numWords) {
        inputColumn(j) = rng.nextFloat()
      }
      inputWeights(i) = normalize(inputColumn)
    }

    for(i <- 0 until outputWeights.size) {
      val outputColumn = new Array[Float](hiddenSize)
      for(j <- 0 until hiddenSize) outputColumn(j) = rng.nextFloat()
      outputWeights(i) = normalize(outputColumn)
    }
  
    (inputWeights, outputWeights)
  }

  // re-weight the outWeights
  def newOutWeights(outWeights : Array[Array[Float]], inVector : Array[Float], error : Array[Float]) {
    for(label <- 0 until outWeights.size) {
      for (neuron <- 0 until outWeights(label).size) outWeights(label)(neuron) += inVector(neuron) * error(label)
    }
  }

  // Multiply the error vector by the new output weights
  def backPropogate(error : Array[Float], outWeights : Array[Array[Float]], inVector : Array[Float]) = {
    val hiddenError = new Array[Float](hiddenSize)
    for(index <- 0 until outWeights.size) {
      for(neuron <- 0 until outWeights(index).size) hiddenError(neuron) += outWeights(index)(neuron) * error(index)
    }
    for(i <- 0 until inVector.size) {
      hiddenError(i) *= inVector(i) * (1f - inVector(i))
    }
    hiddenError
  }

  def newInputWeights(inputWeights : Array[Array[Float]], hiddenError : Array[Float], featureValues : Map[Int, Float]) {
    for(neuron <- 0 until inputWeights.size) {
      featureValues foreach { case(feature, value) => inputWeights(neuron)(feature) += hiddenError(neuron) * featureValues(feature)}
    }
  }

  // set the neural network values globally
  var inputWeights : Array[Array[Float]] = null
  var outputWeights : Array[Array[Float]] = null

  def trainNN(trainData : List[Map[String, String]], wordMap : Map[String, Map[String, Float]]) {
    val trainVectors = vectorize(scala.util.Random.shuffle(trainData))    
   randomInit(wordMap.keys)

    val learningRate = .3f
    for(i <- 0 until 10) {
      trainVectors foreach { case(features, labels) =>
        val inVector = vectorSigmoid(vectorMatrixMultiply(features, inputWeights))
        val outVector = vectorSigmoid(vectorMatrixMultiply(inVector, outputWeights))
        val error = costFunction(outVector, labels)
        newOutWeights(outputWeights, inVector, error)
        val hiddenError = backPropogate(error, outputWeights, inVector)
        newInputWeights(inputWeights, hiddenError, features)
      }
    }
  }

  override 
  def train(trainData : List[Map[String, String]]) {
    wordMap = trainData.flatMap(datum => bagger(datum))
    .groupBy(_._1)
    .map(kv =>(kv._1, kv._2 map (x => x._2)))
    .map(kv => (kv._1, kv._2 groupBy identity mapValues(_.size.toFloat)))
    val totals = wordMap map (kv => (kv._1, kv._2.values.reduce(_+_)))
    wordMap = wordMap map (kv => (kv._1, kv._2 map (xy => (xy._1, xy._2.toFloat / totals.get(kv._1).get))))
    trainNN(trainData, wordMap)
    wordMap
  }

  var ccount = 0
  
  override 
  def classify(tweet: String) : String = {
    val features = classifyVector(cleaner(tweet))
    //features foreach {x => println(x._1 + " " + x._2)}
    val inVector = vectorSigmoid(vectorMatrixMultiply(features, inputWeights))
    val outVector = vectorSigmoid(vectorMatrixMultiply(inVector, outputWeights))
    var min = -1e6f
    var min_index = -1
    for(index <- 0 until outVector.size) {
      if (outVector(index) > min) {
        min = outVector(index)
        min_index = index
      }
    }
    /*
    inVector foreach(x => print(x + " "))
    println()
    outVector foreach(x => print(x + " "))
    println()
    if(ccount > 5) System.exit(0)
    else ccount += 1
    */
    println(min_index + ": " + labelIndex.getOrElse(min_index, "no-non-zero-label-#"))
    labelIndex.getOrElse(min_index, "no-non-zero-label-#")
  }
}
