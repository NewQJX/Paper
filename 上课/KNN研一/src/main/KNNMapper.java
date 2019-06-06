package main;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import com.sun.jersey.api.MessageException;

import util.Distance;

/*
 * map阶段的输入为训练集文本文件格式为：	1.0 2.0 3.0 1		最后一列为类别标签
							1.0 2.1 3.1 1
							
	输出为：(测试样例， DistanceAndLabel)		DistanceAndLabel包含了距离和类别
 */
public class KNNMapper extends Mapper<LongWritable, Text, Instance, DistanceAndLabel> {

	private ArrayList<Instance> testSet = new ArrayList<Instance>(); // 测试集
	// k1为训练集中行偏移量 v1为偏移量对应的内容（属性，类别）

	@Override
	protected void map(LongWritable k1, Text v1, Context context) throws IOException, InterruptedException {
		Instance trainInstance = new Instance(v1.toString()); // 一个训练集中的样例
		double label = trainInstance.getLable(); // 获取训练样例类别

		// 遍历测试集，计算到训练样例的距离
		for (Instance testInstance : testSet) {
			// 计算距离
			double dis = 0.0;
			try {
				dis = Distance.EuclideanDistance(testInstance.getAtrributeValue(), trainInstance.getAtrributeValue());	//计算测试集样例与训练集样例的欧式距离
				DistanceAndLabel dal = new DistanceAndLabel(dis, label);	//将计算出的距离和训练集样例对应的类别标签封装成DistanceAndLabel对象
				context.write(testInstance, dal);	//map输出
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	// 在每个map节点加载测试集   格式：	3.3 6.9 8.8 -1				最后一列为类别标签，都初始化为-1
//								2.5 3.3 10.0 -1
	//将测试集中的每一行都封装成一个Instance对象，加入到testSet集合中
	@Override
	protected void setup(Context context) throws IOException, InterruptedException {
		Path[] testFile = DistributedCache.getLocalCacheFiles(context.getConfiguration());	//获取输入的测试集文件路径
		BufferedReader br = null;		//声明一个字符输入流
		String line;
		for (int i = 0; i < testFile.length; i++) {
			br = new BufferedReader(new FileReader(testFile[0].toString()));		//根据测试集文件路径初始化字符输入流
			while ((line = br.readLine()) != null) {		//按行读取测试集文件
				Instance testInstance = new Instance(line);		//根据测试集的一行，创建一个Instance对象
				testSet.add(testInstance);				//加入到testSet集合中
			}
		}

	}

}
