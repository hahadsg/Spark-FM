package org.apache.spark.ml.regression

import org.apache.hadoop.fs.Path
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.BLAS._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.ml.{PredictionModel, Predictor, PredictorParams}
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.mllib.linalg.{Vector => oldVector, Vectors => oldVectors}
import org.apache.spark.mllib.optimization.{Gradient, GradientDescent, SquaredL2Updater}
import org.apache.spark.mllib.regression.{LabeledPoint => oldLabeledPoint}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.storage.StorageLevel
import breeze.linalg.{axpy => brzAxpy, norm => brzNorm, Vector => BV}

import scala.util.Random


private[regression] trait FactorizationMachinesWithSGDParams
  extends PredictorParams
  with HasMaxIter with HasStepSize with HasTol {

  /**
   * Param for dimensionality of the factorizaion
   * @group getParam
   */
  final val numFactors: IntParam = new IntParam(this, "numFactors", "dimensionality of the factorization")
  final def getNumFactors: Int = $(numFactors)

  /**
    * Param for whether to fit global bias term
    * @group getParam
    */
  final val fitBias: BooleanParam = new BooleanParam(this, "fitBias", "whether to fit global bias term")
  final def getFitBias: Boolean = $(fitBias)

  /**
    * Param for whether to fit linear term (aka 1-way term)
    * @group getParam
    */
  final val fitLinear: BooleanParam = new BooleanParam(this, "fitLinear", "whether to fit linear term (aka 1-way term)")
  final def getFitLinear: Boolean = $(fitLinear)

  /**
    * Param for L2 regularization parameter
    * @group getParam
    */
  final val regParam: DoubleParam = new DoubleParam(this, "regParam", "regularization for L2")
  final def getRegParam: Double = $(regParam)

  /**
    * Param for mini-batch fraction, must be in range (0, 1]
    * @group getParam
    */
  final val miniBatchFraction: DoubleParam = new DoubleParam(this, "miniBatchFraction", "mini-batch fraction")
  final def getMiniBatchFraction: Double = $(miniBatchFraction)

  /**
    * Param for standard deviation of initial coefficients
    * @group getParam
    */
  final val initStd: DoubleParam = new DoubleParam(this, "initStd", "standard deviation of initial coefficients")
  final def getInitStd: Double = $(initStd)

  /**
    * Param for loss function type
    * @group getParam
    */
  final val lossFunc: Param[String] = new Param[String](this, "lossFunc", "loss function type")
  final def getLossFunc: String = $(lossFunc)

}

class FactorizationMachinesWithSGD (
    override val uid: String)
  extends Predictor[Vector, FactorizationMachinesWithSGD, FactorizationMachinesWithSGDModel]
  with FactorizationMachinesWithSGDParams with DefaultParamsWritable with Logging {

  def this() = this(Identifiable.randomUID("fm"))

  /**
    * Set the dimensionality of the factorizaion
    * Default is 8
    *
    * @group setParam
    */
  def setNumFactors(value: Int): this.type = set(numFactors, value)
  setDefault(numFactors -> 8)

  /**
    * Set whether to fit global bias term
    * Default is true
    *
    * @group setParam
    */
  def setFitBias(value: Boolean): this.type = set(fitBias, value)
  setDefault(fitBias -> true)

  /**
    * Set whether to fit linear term
    * Default is true
    *
    * @group setParam
    */
  def setFitLinear(value: Boolean): this.type = set(fitLinear, value)
  setDefault(fitLinear -> true)

  /**
    * Set the L2 regularization parameter
    * Default is 0.0
    *
    * @group setParam
    */
  def setRegParam(value: Double): this.type = {
    require(value >= 0.0, s"Regularization parameter must be greater than 0.0 but got $value")
    set(regParam, value)
  }
  setDefault(regParam -> 0.0)

  /**
    * Set the mini-batch fraction parameter
    * Default is 1.0
    *
    * @group setParam
    */
  def setMiniBatchFraction(value: Double): this.type = {
    require(value > 0 && value <= 1.0,
      s"Fraction for mini-batch SGD must be in range (0, 1] but got $value")
    set(miniBatchFraction, value)
  }
  setDefault(miniBatchFraction -> 1.0)

  /**
    * Set the standard deviation of initial coefficients
    * Default is 0.01
    *
    * @group setParam
    */
  def setInitStd(value: Double): this.type = set(initStd, value)
  setDefault(initStd -> 0.01)

  /**
    * Set the maximum number of iterations
    * Default is 100
    *
    * @group setParam
    */
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  /**
    * Set the initial step size for the first step (like learning rate)
    * Default is 0.1
    *
    * @group setParam
    */
  def setStepSize(value: Double): this.type = set(stepSize, value)
  setDefault(stepSize -> 0.1)

  /**
    * Set the convergence tolerance of iterations
    * Default is 1E-8
    *
    * @group setParam
    */
  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-8)

  /**
    * Set the loss function type, only support logistic loss now.
    * Default is logisticloss
    *
    * @group setParam
    */
  def setLossFunc(value: String): this.type = set(lossFunc, value)
  setDefault(lossFunc -> "logisticloss")

  override protected[spark] def train(dataset: Dataset[_]): FactorizationMachinesWithSGDModel = {
    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    train(dataset, handlePersistence)
  }

  protected[spark] def train(
      dataset: Dataset[_],
      handlePersistence: Boolean): FactorizationMachinesWithSGDModel = {
    val instances: RDD[oldLabeledPoint] =
      dataset.select(col($(labelCol)), col($(featuresCol))).rdd.map {
        case Row(label: Double, features: Vector) =>
          oldLabeledPoint(label, features)
      }

    if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)

    val instr = Instrumentation.create(this, instances)
    instr.logParams(numFactors, fitBias, fitLinear, regParam,
      miniBatchFraction, maxIter, stepSize, tol, lossFunc)

    val numFeatures = instances.first().features.size
    instr.logNumFeatures(numFeatures)

    // initialize coefficients
    val coefficientsSize = $(numFactors) * numFeatures +
      (if ($(fitLinear)) numFeatures else 0) +
      (if ($(fitBias)) 1 else 0)
    val initialCoefficients = Vectors.dense(Array.fill(coefficientsSize)(Random.nextGaussian() * $(initStd)))

    val data = instances.map{ case oldLabeledPoint(label, features) => (label, features) }

    // optimize coefficients with gradient descent
    val gradient = BaseFactorizationMachinesWithSGDGradient.parseLossFuncStr(
      $(lossFunc), $(numFactors), $(fitBias), $(fitLinear), numFeatures)
    val updater = new SquaredL2Updater()
    val optimizer = new GradientDescent(gradient, updater)
      .setStepSize($(stepSize))
      .setNumIterations($(maxIter))
      .setRegParam($(regParam))
      .setMiniBatchFraction($(miniBatchFraction))
      .setConvergenceTol($(tol))
    val coefficients = optimizer.optimize(data, initialCoefficients)

    if (handlePersistence) instances.unpersist()

    val model = copyValues(new FactorizationMachinesWithSGDModel(uid, coefficients, numFeatures))
    model
  }

  override def copy(extra: ParamMap): FactorizationMachinesWithSGD = defaultCopy(extra)
}

object FactorizationMachinesWithSGD extends DefaultParamsReadable[FactorizationMachinesWithSGD] {
  override def load(path: String): FactorizationMachinesWithSGD = super.load(path)
}

class FactorizationMachinesWithSGDModel (
    override val uid: String,
    val coefficients: oldVector,
    override val numFeatures: Int)
  extends PredictionModel[Vector, FactorizationMachinesWithSGDModel]
  with FactorizationMachinesWithSGDParams with MLWritable {

  /**
    * Returns Factorization Machines coefficients
    * coefficients concat from 2-way coefficients, 1-way coefficients, global bias
    * index 0 ~ numFeatures * numFactors are 2-way coefficients
    * Following indices are 1-way coefficients and global bias.
    *
    * @return Vector
    */
  def getCoefficients: Vector = coefficients

  private lazy val gradient = BaseFactorizationMachinesWithSGDGradient.parseLossFuncStr(
    $(lossFunc), $(numFactors), $(fitBias), $(fitLinear), numFeatures)

  override protected def predict(features: Vector): Double = {
    val rawPrediction = gradient.getRawPrediction(features, coefficients)
    gradient.getPrediction(rawPrediction)
  }

  override def copy(extra: ParamMap): FactorizationMachinesWithSGDModel = {
    copyValues(new FactorizationMachinesWithSGDModel(
      uid, coefficients, numFeatures), extra)
  }

  override def write: MLWriter = new FactorizationMachinesWithSGDModel.FactorizationMachinesWithSGDModelWriter(this)
}

object FactorizationMachinesWithSGDModel extends MLReadable[FactorizationMachinesWithSGDModel] {

  override def read: MLReader[FactorizationMachinesWithSGDModel] = new FactorizationMachinesWithSGDModelReader

  override def load(path: String): FactorizationMachinesWithSGDModel = super.load(path)

  private[FactorizationMachinesWithSGDModel] class FactorizationMachinesWithSGDModelWriter(
      instance: FactorizationMachinesWithSGDModel) extends MLWriter with Logging {

    private case class Data(
        numFeatures: Int,
        coefficients: oldVector)

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.numFeatures, instance.coefficients)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class FactorizationMachinesWithSGDModelReader extends MLReader[FactorizationMachinesWithSGDModel] {

    private val className = classOf[FactorizationMachinesWithSGDModel].getName

    override def load(path: String): FactorizationMachinesWithSGDModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.format("parquet").load(dataPath)

      val Row(numFeatures: Int, coefficients: oldVector) = data
        .select("numFeatures", "coefficients").head()
      val model = new FactorizationMachinesWithSGDModel(
        metadata.uid, coefficients, numFeatures)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }
}

private[ml] abstract class BaseFactorizationMachinesWithSGDGradient(
    numFactors: Int,
    fitBias: Boolean,
    fitLinear: Boolean,
    numFeatures: Int) extends Gradient {

  override def compute(
      data: oldVector,
      label: Double,
      weights: oldVector,
      cumGradient: oldVector): Double = {
    val rawPrediction = getRawPrediction(data, weights)
    val rawGradient = getRawGradient(data, weights)
    val multiplier = getMultiplier(rawPrediction, label)
    axpy(multiplier, rawGradient, cumGradient)
    val loss = getLoss(rawPrediction, label)
    loss
  }

  def getPrediction(rawPrediction: Double): Double

  protected def getMultiplier(rawPrediction: Double, label: Double): Double

  protected def getLoss(rawPrediction: Double, label: Double): Double

  private val sumVX = Array.fill(numFactors)(0.0)

  def getRawPrediction(data: oldVector, weights: oldVector): Double = {
    var rawPrediction = 0.0
    val vWeightsSize = numFeatures * numFactors

    if (fitBias) rawPrediction += weights(weights.size - 1)
    if (fitLinear) {
      data.foreachActive { case (index, value) =>
        rawPrediction += weights(vWeightsSize + index) * value
      }
    }
    (0 until numFactors).foreach { f =>
      var sumSquare = 0.0
      var squareSum = 0.0
      data.foreachActive { case (index, value) =>
        val vx = weights(index * numFactors + f) * value
        sumSquare += vx * vx
        squareSum += vx
      }
      sumVX(f) = squareSum
      squareSum = squareSum * squareSum
      rawPrediction += 0.5 * (squareSum - sumSquare)
    }

    rawPrediction
  }

  private def getRawGradient(data: oldVector, weights: oldVector): oldVector = {
    val gradient = Array.fill(weights.size)(0.0)
    val vWeightsSize = numFeatures * numFactors

    if (fitBias) gradient(weights.size - 1) += 1.0
    if (fitLinear) {
      data.foreachActive { case (index, value) =>
        gradient(vWeightsSize + index) += value
      }
    }
    (0 until numFactors).foreach { f =>
      data.foreachActive { case (index, value) =>
        gradient(index * numFactors + f) +=
          value * sumVX(f) - weights(index * numFactors + f) * value * value
      }
    }

    oldVectors.dense(gradient)
  }
}

object BaseFactorizationMachinesWithSGDGradient {

  def parseLossFuncStr(
      lossFunc: String,
      numFactors: Int,
      fitBias: Boolean,
      fitLinear: Boolean,
      numFeatures: Int): BaseFactorizationMachinesWithSGDGradient = {
    lossFunc match {
      case "logisticloss" => new LogisticFactorizationMachinesWithSGDGradient(numFactors, fitBias, fitLinear, numFeatures)
      case _ => throw new IllegalArgumentException(s"loss function type $lossFunc is invalidation")
    }
  }
}

private[ml] class LogisticFactorizationMachinesWithSGDGradient(
    numFactors: Int,
    fitBias: Boolean,
    fitLinear: Boolean,
    numFeatures: Int)
  extends BaseFactorizationMachinesWithSGDGradient(
    numFactors: Int,
    fitBias: Boolean,
    fitLinear: Boolean,
    numFeatures: Int) {

  override def getPrediction(rawPrediction: Double): Double = {
    1.0 / (1.0 + math.exp(-rawPrediction))
  }

  override protected def getMultiplier(rawPrediction: Double, label: Double): Double = {
    val prediction = getPrediction(rawPrediction)
    val multiplier = prediction - label
    multiplier
  }

  override protected def getLoss(rawPrediction: Double, label: Double): Double = {
    if (label > 0) MLUtils.log1pExp(-rawPrediction)
    else MLUtils.log1pExp(rawPrediction)
  }
}

