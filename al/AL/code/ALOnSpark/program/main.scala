import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.storage.StorageLevel
import java.util
import scala.collection.JavaConverters._

/*
* parameter1：有类标数据存放路径
* parameter2：无类标数据存放路径
* parameter3：更新有类标数据存放存放路径
* parameter4：隐层节点数
* parameter5：循环次数
* parameter6：每次选取的样例个数
* parameter7：elm输出节点个数
* */

object main {

  def main(args: Array[String]) = {

    val conf = new SparkConf().setAppName("ELM_OnSpark")
    conf.set("spark.default.parallelism","60")
    conf.set("spark.memory.useLegacyMode","true")
    //超过这个限制，shuffle溢出数据将会保存到磁盘上
    conf.set("spark.shuffle.memoryFraction","0.3")
    conf.set("spark.storage.memoryFraction","0.5")
    //合并的中间文件将会被创建
    conf.set("spark.shuffle.consolidateFiles","true")
    conf.set("spark.executor.memory","2G")
    conf.set("spark.driver.memory","2G")
    conf.set("spark.driver.memoryOverhead","2048M")
    conf.set("spark.executor.memoryOverhead","2048M")
    conf.set("spark.executor.cores","4")
    conf.set("spark.files.fetchTimeout","300s")
    conf.set("spark.network.timeout","300s")
    conf.set("spark.rpc.numRetries","10")
    val sc = new SparkContext(conf)

    //导入有类标数据集
    val trainHasLabelInputData = sc.textFile(args(0), 1).persist(StorageLevel.MEMORY_AND_DISK)
    val trainHasLabelInputDataRDD = trainHasLabelInputData.map(line => {
      val array = line.split(" ")
      val attribute = new Array[Double](array.length)
      for (i <- 0 until array.length) {
        attribute(i) = array(i).toDouble
      }
      attribute
    }).persist(StorageLevel.MEMORY_AND_DISK)

    //导入无类标数据集
    val trainHideLabelInputData = sc.textFile(args(1), 3).persist(StorageLevel.MEMORY_AND_DISK)
    val trainHideLabelInputDataLineRDD = trainHideLabelInputData.map(line => {
      val array = line.split(" ")
      val attribute = new Array[Double](array.length)
      for (i <- 0 until array.length) {
        attribute(i) = array(i).toDouble
      }
      attribute
    }).persist(StorageLevel.MEMORY_AND_DISK)

    //计算有类标数据集样例个数
    val trainHasLabelList = trainHasLabelInputDataRDD.take(trainHasLabelInputDataRDD.count().toInt)
    //广播有类标数据集
    var trainHasLabelList1 = sc.broadcast(trainHasLabelList)

    //计算无类标数据集熵值，并选出前L个熵值大的样例
    def compute_entry(iter: Iterator[Array[Double]]): Iterator[Array[Double]] = {
      var result = new util.HashMap[Double, Array[Double]]()
      var real = new util.HashMap[Double, Array[Double]]()
      val e = new elm(1, args(3).toInt, "sig")
      e.train(trainHasLabelList1.value, args(6).toInt)
      println(e.getTrainingAccuracy)
      println(e.getTrainingTime)

      while (iter.hasNext) {
        val sample = iter.next()
        val entry = e.testHidenLabelOut(sample, args(6).toInt)
        result.put(entry, sample)
      }
      val key = result.keySet().toArray()
      util.Arrays.sort(key)
      for (i <- key.toList.size - args(5).toInt until key.toList.size) {
        var keyy = key.toList(i).toString.toDouble
        real.put(keyy, result.get(keyy))
      }
      val it = real.values().iterator().asScala
      it
    }

    //计算无类标数据集熵值，选出前L个熵值大的样例放入到有类标数据集
    val resulttrainHideLabelRDD = trainHideLabelInputDataLineRDD.mapPartitions(compute_entry).persist(StorageLevel.MEMORY_AND_DISK)
    var updateTrainHasLabelInputDataLineRDD = trainHasLabelInputDataRDD.map(_.toList).union(resulttrainHideLabelRDD.map(_.toList)).persist(StorageLevel.MEMORY_AND_DISK)

    //将更新后的有类标训练数据集保存成文件
    val reHasLabel = updateTrainHasLabelInputDataLineRDD.map(line => {
      var result = new Array[Double](line.size)
      for (i <- 0 until line.size) {
        result(i) = line(i)
      }
      result.toList
    }).persist(StorageLevel.MEMORY_AND_DISK).repartition(1).saveAsTextFile((args(2)))

    for (i <- 0 until args(4).toInt) {
      val TrainHasLabelInputDataAttr = updateTrainHasLabelInputDataLineRDD.map(line => {
        val attribute = new Array[Double](line.length)
        for (i <- 0 until line.length) {
          attribute(i) = line(i)
        }
        attribute
      }).persist(StorageLevel.MEMORY_AND_DISK)

      val trainHasLabelList = TrainHasLabelInputDataAttr.take(TrainHasLabelInputDataAttr.count().toInt)
      trainHasLabelList1.unpersist()
      trainHasLabelList1 = sc.broadcast(trainHasLabelList)

      //计算更新后无类标数据集熵值，选出前L个熵值大的样例放入到有类标数据集
      val updateResulttrainHideLabelOutRDD = trainHideLabelInputDataLineRDD.mapPartitions(compute_entry).persist(StorageLevel.MEMORY_AND_DISK)
      val updateTrainHasLabelInputDataLineRDD1 = updateTrainHasLabelInputDataLineRDD.map(_.toList).union(updateResulttrainHideLabelOutRDD.map(_.toList)).persist(StorageLevel.MEMORY_AND_DISK)
      updateTrainHasLabelInputDataLineRDD = updateTrainHasLabelInputDataLineRDD1.persist(StorageLevel.MEMORY_AND_DISK)

      //将再次更新后的有类标训练数据集保存成文件
      val reHasLabel = updateTrainHasLabelInputDataLineRDD1.map(line => {
        var result = new Array[Double](line.size)
        for (i <- 0 until line.size) {
          result(i) = line(i)
        }
        result.toList
      }).persist(StorageLevel.MEMORY_AND_DISK).repartition(numPartitions = 1).saveAsTextFile((args(2) + i.toString))
    }
    sc.stop()
  }
}
