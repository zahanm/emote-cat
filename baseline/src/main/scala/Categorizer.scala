package main.scala

/**
 * Template Classifier class
 *
 * @author gibbons4
 */
abstract class Classifier {
  def train(trainData : List[Map[String, String]]) 
  def classify(tweet: String) : String
}
