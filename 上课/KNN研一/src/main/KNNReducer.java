package main;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.collections.IteratorUtils;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import util.Sort;

//testSample	DistAndLabel(dist,lable)
public class KNNReducer extends Reducer<Instance, DistanceAndLabel, Text, NullWritable> {
	private int k; // 声明参数k

	@Override
	protected void setup(Context context) throws IOException, InterruptedException {
		k = context.getConfiguration().getInt("k", 1); // 获取k的值
	}

	@Override
	protected void reduce(Instance k3, Iterable<DistanceAndLabel> v3, Context context)
			throws IOException, InterruptedException {

		ArrayList<DistanceAndLabel> list = new ArrayList<>(); // 初始化一个集合，保存输入的value值

		for (DistanceAndLabel d : v3) { // 将v3中的元素，加入到list集合中
			DistanceAndLabel tmp = new DistanceAndLabel(d.distance, d.label);
			list.add(tmp);
		}

		list = Sort.getNearest(list, k); // 在该测试样例与训练样例的距离中，找到最近的k个距离，和对应的类别标签
		try {
			Double label = valueOfMostFrequent(list); // 投票找到最多的类别标签
			Instance ins = new Instance(k3.getAtrributeValue(), label); // 将测试样例，和投票得到的类别封装成Instance对象
			context.write(new Text(ins.toString()), NullWritable.get()); // 输出：测试样例和预测的结果
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	// 在list集合中找到出现次数最多的类别
	public Double valueOfMostFrequent(ArrayList<DistanceAndLabel> list) throws Exception {
		if (list.isEmpty())
			throw new Exception("list is empty!");
		else {
			HashMap<Double, Integer> tmp = new HashMap<Double, Integer>();
			for (int i = 0; i < list.size(); i++) {
				if (tmp.containsKey(list.get(i).label)) {
					Integer frequence = tmp.get(list.get(i).label) + 1;
					tmp.remove(list.get(i).label);
					tmp.put(list.get(i).label, frequence);
				} else {
					tmp.put(list.get(i).label, new Integer(1));
				}
			}
			// find the value with the maximum frequence.
			Double value = new Double(0.0);
			Integer frequence = new Integer(Integer.MIN_VALUE);
			Iterator<Entry<Double, Integer>> iter = tmp.entrySet().iterator();
			while (iter.hasNext()) {
				Map.Entry<Double, Integer> entry = (Map.Entry<Double, Integer>) iter.next();
				if (entry.getValue() > frequence) {
					frequence = entry.getValue();
					value = entry.getKey();
				}
			}
			return value;
		}
	}

}
