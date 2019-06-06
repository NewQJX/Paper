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
	private int k; // ��������k

	@Override
	protected void setup(Context context) throws IOException, InterruptedException {
		k = context.getConfiguration().getInt("k", 1); // ��ȡk��ֵ
	}

	@Override
	protected void reduce(Instance k3, Iterable<DistanceAndLabel> v3, Context context)
			throws IOException, InterruptedException {

		ArrayList<DistanceAndLabel> list = new ArrayList<>(); // ��ʼ��һ�����ϣ����������valueֵ

		for (DistanceAndLabel d : v3) { // ��v3�е�Ԫ�أ����뵽list������
			DistanceAndLabel tmp = new DistanceAndLabel(d.distance, d.label);
			list.add(tmp);
		}

		list = Sort.getNearest(list, k); // �ڸò���������ѵ�������ľ����У��ҵ������k�����룬�Ͷ�Ӧ������ǩ
		try {
			Double label = valueOfMostFrequent(list); // ͶƱ�ҵ���������ǩ
			Instance ins = new Instance(k3.getAtrributeValue(), label); // ��������������ͶƱ�õ�������װ��Instance����
			context.write(new Text(ins.toString()), NullWritable.get()); // ���������������Ԥ��Ľ��
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	// ��list�������ҵ����ִ����������
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
