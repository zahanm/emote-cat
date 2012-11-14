
name := "clustering"

version := "0.1"

libraryDependencies ++= Seq(
  "org.apache.mahout" % "mahout-core" % "0.7",
  "org.scalanlp" %% "breeze-math" % "0.1",
  "org.scalanlp" %% "breeze-learn" % "0.1",
  "org.scalanlp" %% "breeze-process" % "0.1",
  "org.scalanlp" %% "breeze-viz" % "0.1"
)

resolvers ++= Seq(
   "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/"
)
