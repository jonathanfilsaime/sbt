import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.{Row, SparkSession}


object Hello {

  def main(args: Array[String]) = {

    System.setProperty("hadoop.home.dir", "C:\\hadoop")

    val spark = SparkSession
      .builder()
      .appName("Twitter classification Pipeline")
      .config("spark.master", "local")
      .getOrCreate()

    val tweetCSV = spark.read
      .textFile("./Tweets.csv")
      .rdd

    val header = tweetCSV.first()
    val tweetCSVWithoutheader = tweetCSV.filter( x => x !=header)
    val data = tweetCSVWithoutheader.map{ x =>
      val column = x.split(",")
      if (column(0).matches("^(0|[1-9][0-9]*)$")
        && !column(1).isEmpty
        && !(assign(column(1)).equals(9))
        && !column(0).equals("569851578276048896")
        && !column(0).equals("569473998519578624")
        && !column(0).equals("567800574051151872")
        && !column(0).equals("568182544124014592")
        && !column(0).equals("570139793608175616")
        && !column(0).equals("569047438880841728")
        && !column(0).equals("568757671819661314")
        && !column(0).equals("568637541513089024"))
      {
        (column(0), column(10), assign(column(1)))
      } else {
        (null, null, 7)
      }
    }

    val cleanData = data.filter(x => x._1 != null || x._2 != null || x._3 != 7)

    cleanData.foreach(x => println("-------> " + x))

    val num = cleanData.count()
    println(num)

    val trainingPercent = (num * .8).toInt
    println(trainingPercent)

    val twitterDataFrame = spark.createDataFrame(cleanData.take(trainingPercent)).toDF("id", "text", "label")
    println(twitterDataFrame.count())

//     Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, lr))

    // Fit the pipeline to training documents.
    val model = pipeline.fit(twitterDataFrame)

    // Now we can optionally save the fitted pipeline to disk
    model.write.overwrite().save("./tmp/spark-logistic-regression-model")

    // We can also save this unfit pipeline to disk
    pipeline.write.overwrite().save("./tmp/unfit-lr-model")

    // And load it back in during production
//    val sameModel = PipelineModel.load("/tmp/spark-logistic-regression-model")

    val cleanDataWithoutLabel = cleanData.map(x => (x._1, x._2))

    val twitterTestDataFrame = spark.createDataFrame(cleanDataWithoutLabel).toDF("id", "text")
    println(twitterTestDataFrame.count())

    val prediction = model.transform(twitterTestDataFrame)
      .select("id", "text", "probability", "prediction")
      .collect()

    prediction.foreach(println)

  }

  def assign(value : Any): Int   = {

    if(value != null)
    {
      if("negative".equalsIgnoreCase(value.toString))
      {
        return  0
      }
      else if ("positive".equalsIgnoreCase(value.toString))
      {
        return 1
      }
      else
      {
        return 9
      }

    }
    else
    {
      return 0
    }
  }
}