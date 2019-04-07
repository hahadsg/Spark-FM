name := "spark-ml"

version := "0.1"

scalaVersion := "2.12.8"

val sparkVersion = "2.4.0"

libraryDependencies := Seq(
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided"
)
