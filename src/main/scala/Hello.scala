import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, StringIndexer, StringIndexerModel, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

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

    val cleanData = dataCleanUp(tweetCSV)

    val num = cleanData.count()
    println("number of total rows: " + num)

    val trainingPercent = (num * .8).toInt
    println(trainingPercent)

    //training data selection 80%
    val twitterDataFrame = spark.createDataFrame(cleanData.take(trainingPercent)).toDF("id", "text", "stringLabel")
    println("number of training data rows: " + twitterDataFrame.count())

    //transforming string label to number
    val twitterDataFrameIndexed = StringToNumberIndexer(twitterDataFrame)

    //Configure an ML pipeline, which consists of four stages: tokenizer, StopWordsRemover, hashingTF, and lr.
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered")
    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(remover.getOutputCol)
      .setOutputCol("features")
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, hashingTF, lr))

    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)

    //fit the pipeline to training documents.
    val model = pipeline.fit(twitterDataFrameIndexed)

    //test data selection 20%
    val trainCleanData = cleanData.take(trainingPercent)
    val testData = cleanData.filter( x => !trainCleanData.contains(x))
    val cleanDataWithoutLabel = testData.map(x => (x._1, x._2))

    val twitterTestDataFrame = spark.createDataFrame(testData).toDF("id", "text", "stringLabel")
    println("number of test data rows: " + twitterTestDataFrame.count())

    val twitterTestWithoutLabelDataFrame = spark.createDataFrame(cleanDataWithoutLabel).toDF("id", "text")
    println("number of test data rows: " + twitterTestWithoutLabelDataFrame.count())

    val prediction = model.transform(twitterTestWithoutLabelDataFrame)

    prediction.rdd.saveAsTextFile("./prediction")

  }

  private def StringToNumberIndexer(twitterDataFrame: DataFrame) = {
    val indexer = new StringIndexer()
      .setInputCol("stringLabel")
      .setOutputCol("label")
      .fit(twitterDataFrame)
    indexer.transform(twitterDataFrame)
  }

  private def dataCleanUp(tweetCSV: RDD[String]) = {

    val header = tweetCSV.first()
    val tweetCSVWithoutheader = tweetCSV.filter(x => x != header)

    val data = tweetCSVWithoutheader.map { x =>
      val column = x.split(",")
      if (column(0).matches("^(0|[1-9][0-9]*)$")
        && !column(1).isEmpty
        && !column(0).equals("569851578276048896")
        && !column(0).equals("569473998519578624")
        && !column(0).equals("567800574051151872")
        && !column(0).equals("568182544124014592")
        && !column(0).equals("570139793608175616")
        && !column(0).equals("569047438880841728")
        && !column(0).equals("568757671819661314")
        && !column(0).equals("568637541513089024"))
      {
        (column(0), column(10), column(1))
      }
      else
      {
        (null, null, null)
      }
    }

    val cleanData = data.filter(x => x._1 != null || x._2 != null || x._3 != null)
    cleanData
  }
}