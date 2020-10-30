//import org.apache.spark.sql.SparkSession
//
//object twitterClassificationPipelineMainClass {
//
//  def main(args: Array[String]) = {
//
//    val spark = SparkSession
//      .builder()
//      .appName("Twitter classification Pipeline")
//      .config("spark.master", "local")
//      .getOrCreate()
//
//    val tweetDF = spark.read.load("\sbt\Tweets.csv")
//    tweetDF.take(1);
//  }
//}