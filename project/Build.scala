
import sbt._
import Keys._

object Root extends Build {

  lazy val root = Project(
    id = "root",
    base = file("."),
    aggregate = Seq(clustering, baseline)
  )

  lazy val clustering = Project(
    id = "clustering",
    base = file("clustering")
  )

  lazy val classification = Project(
    id = "classification",
    base = file("classification")
  )

  lazy val baseline = Project(
    id = "baseline",
    base = file("baseline")
  )

}
