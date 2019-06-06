package main;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

import org.apache.commons.collections.IteratorUtils;
import org.apache.hadoop.mapreduce.Reducer;
import org.mockito.internal.util.ArrayUtils;

import com.google.inject.spi.Message;
import com.sun.jersey.api.MessageException;

import util.Sort;


/*
 * combiner阶段获取本地map阶段的输出，将相同测试集样例，和其与每个训练集样例距离和类别，合并到一起作为combiner的输入
 * 输出为：(测试集样例，本地节点计算出的最近的k个近邻)
 */
public class KNNCombiner extends Reducer<Instance, DistanceAndLabel, Instance, DistanceAndLabel> {
	
	private int k;		//声明参数k

	@Override
	protected void setup(Context context) throws IOException, InterruptedException {
		this.k = context.getConfiguration().getInt("k", 1);			//获取k的值
	}
	
	@Override
	protected void reduce(Instance testInstance, Iterable<DistanceAndLabel> iter,Context context)
			throws IOException, InterruptedException {
		
		ArrayList<DistanceAndLabel> list = new ArrayList<>();		//初始化一个集合，保存输入的value值
		
		for(DistanceAndLabel d : iter) {			//将iter中的元素，加入到list集合中
			DistanceAndLabel tmp = new DistanceAndLabel(d.distance, d.label);
			list.add(tmp);
		}
		
		list = Sort.getNearest(list, k);			//在该测试样例与训练样例的距离中，找到最近的k个距离，和对应的类别标签
		for(DistanceAndLabel dal : list) {		//输出	(测试集样例，本地节点计算出的最近的k个近邻)
			context.write(testInstance, dal);
		}

	}

	

	
	
	

}
