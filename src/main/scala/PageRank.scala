import org.apache.spark
import org.apache.spark.sql.SparkSession

object PageRank {
  def main(args: Array[String]): Unit = {
    if (args.length != 3) {
      println("Usage: PageRank input iterations output")
    }
    // Create spark session
    val spark = SparkSession
      .builder()
      .appName("Databricks Spark Example")
      .config("spark.master", "local")
      .getOrCreate()
    // Read data into rdd
    val csvData = spark.read.textFile(args(0)).rdd
    // Remove header from rdd
    val header = csvData.first()
    val csvDataWithoutHeader = csvData.filter(x => x != header)
    // Create edges for each flight (pairs)
    val edges = csvDataWithoutHeader.map { x =>
      val column = x.split(",")
      (column(1).stripPrefix("\"").stripSuffix("\""), column(3).stripPrefix("\"").stripSuffix("\""))
    }
    // Create edges for each key (DFW, (
    val edgesByKey = edges.distinct().groupByKey()
    var ranks = edgesByKey.mapValues(x => 10.0)
    for (i <- 1 to args(1).toInt) {
      ranks = edgesByKey
        .join(ranks)
        .values
        .flatMap { case (airports, rank) => {
          val size = airports.size
          airports.map(airport => (airport, rank / size))
        }
        }
        .reduceByKey(_ + _).mapValues(0.15 + 0.85 * _)
    }
    ranks.sortBy(_._2, false).saveAsTextFile(args(2))
  }
}