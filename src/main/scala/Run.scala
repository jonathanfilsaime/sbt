import java.io.File

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

import scala.reflect.io.Directory

object Run {

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
    val rawData = tweetCSVWithoutheader.map{ x =>
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

    val cleanData = rawData.filter(x => x._1 != null || x._2 != null || x._3 != 7)

    cleanData.foreach(x => println("-------> " + x))

    println("cleanData")
    val num = cleanData.count()
    println(num)

    println("trainingPercent")
    val trainingPercent = (num * .8).toInt
    println(trainingPercent)

    println("twitterDataFrame")
    val twitterDataFrameWithoutId = cleanData.map(x => (x._2, x._3))
    val twitterDataFrame = spark.createDataFrame(twitterDataFrameWithoutId.take(trainingPercent)).toDF( "text", "label")
    println(twitterDataFrame.count())

    println("directory")
    val directory = new Directory(new File("./data/cleandata"))
    directory.deleteRecursively()

    //write the file to disk
    twitterDataFrame.rdd.map(_.toString()).saveAsTextFile("./data/cleandata")

    println("problem1")
    // Load training data in LIBSVM format
    val data = MLUtils.loadLibSVMFile(spark.sparkContext, "./data/cleandata")

    println("problem2")
    // Split data into training (60%) and test (40%)
    val Array(training, test) = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    training.cache()

    // Run training algorithm to build the model
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(training)

    // Clear the prediction threshold so the model will return probabilities
    model.clearThreshold

    // Compute raw scores on the test set
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    // Instantiate metrics object
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    // Precision by threshold
    val precision = metrics.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }

    // Recall by threshold
    val recall = metrics.recallByThreshold
    recall.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }

    // Precision-Recall Curve
    val PRC = metrics.pr

    // F-measure
    val f1Score = metrics.fMeasureByThreshold
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 1")
    }

    val beta = 0.5
    val fScore = metrics.fMeasureByThreshold(beta)
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 0.5")
    }

    // AUPRC
    val auPRC = metrics.areaUnderPR
    println("Area under precision-recall curve = " + auPRC)

    // Compute thresholds used in ROC and PR curves
    val thresholds = precision.map(_._1)

    // ROC Curve
    val roc = metrics.roc

    // AUROC
    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)

//    //     Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
//    val tokenizer = new Tokenizer()
//      .setInputCol("text")
//      .setOutputCol("words")
//    val hashingTF = new HashingTF()
//      .setNumFeatures(1000)
//      .setInputCol(tokenizer.getOutputCol)
//      .setOutputCol("features")
//    val lr = new LogisticRegression()
//      .setMaxIter(10)
//      .setRegParam(0.001)
//    val pipeline = new Pipeline()
//      .setStages(Array(tokenizer, hashingTF, lr))
//
//    // Fit the pipeline to training documents.
//    val model = pipeline.fit(twitterDataFrame)
//
//    // Now we can optionally save the fitted pipeline to disk
//    model.write.overwrite().save("./tmp/spark-logistic-regression-model")
//
//    // We can also save this unfit pipeline to disk
//    pipeline.write.overwrite().save("./tmp/unfit-lr-model")
//
//    // And load it back in during production
//    val sameModel = PipelineModel.load("./tmp/spark-logistic-regression-model")
//
//    val cleanDataWithoutLabel = cleanData.map(x => (x._1, x._2))
//
//    val twitterTestDataFrame = spark.createDataFrame(cleanDataWithoutLabel).toDF("id", "text")
//    println(twitterTestDataFrame.count())
//
//    val prediction = model.transform(twitterTestDataFrame)
//      .select("id", "text", "probability", "prediction")
//      .collect()
//
//    prediction.foreach(println)

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
