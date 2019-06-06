package main;


import java.net.URI;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.filecache.DistributedCache;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class KNNMain {

	public static void main(String[] args) throws Exception {
		
		Job job = Job.getInstance();		//创建一个mapreduce任务
		job.setJobName("KNN");			//任务名字为KNN
		DistributedCache.addCacheFile(URI.create(args[2]), job.getConfiguration());		//从命令行获取测试集的文件路径
		job.setJarByClass(KNNMain.class);				//设置任务的主程序
		job.getConfiguration().setInt("k", Integer.parseInt(args[3]));		//从命令行获取k的值
		
		job.setMapperClass(KNNMapper.class);			//设置map阶段的任务
		job.setMapOutputKeyClass(Instance.class);		//设置map阶段的输出key类型
		job.setMapOutputValueClass(DistanceAndLabel.class);		//设置map阶段输出value类型
		
		job.setCombinerClass(KNNCombiner.class);		//设置combiner
		
		job.setReducerClass(KNNReducer.class);		//设置reduce阶段的任务
		job.setOutputKeyClass(Text.class);			//设置reudce阶段输出key类型
		job.setOutputValueClass(NullWritable.class);	//设置reduce阶段输出value类型
		
		FileInputFormat.addInputPath(job, new Path(args[0]));		//训练集文件的路径
		FileOutputFormat.setOutputPath(job, new Path(args[1]));		//任务输出结果路径
		
		job.waitForCompletion(true);		//开始执行任务
	}

}
