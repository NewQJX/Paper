package al;
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


public class ALMain {
										
	private int iterationNum;				//迭代次数	
	private String sourcePath_Train;		//有类标数据				
	private String sourcePath_Test;			//无类标数据				
	private String outputPath;				//选出样例的输出路径（与有类标数据路径保持一致）				
	private Configuration conf;								
	
	public ALMain(int iterationNum, String sourcePath_Train, String sourcePath_Test, String outputPath,
			Configuration conf) {									
		this.iterationNum = iterationNum;						
		this.sourcePath_Test = sourcePath_Test;							
		this.sourcePath_Train = sourcePath_Train;				
		this.outputPath = outputPath;							
		this.conf = conf;										
	}
	
	
	/*
	 *  程序main方法 
	 */
	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
		if(args.length != 7) {
			System.out.println("Usage: NUM_SELECT  NUM_CLASS  hidden_unit  "
					+ "labeledPath  unlabeledPath   labeledPath(outputpath)  iterationNum");
			System.exit(-1);
		}
		System.out.println("-----start-----");
		long startTime = System.currentTimeMillis();
		Configuration conf = new Configuration();
		conf.setInt("NUM_SELECT", Integer.parseInt(args[0]));
		conf.setInt("NUM_CLASS", Integer.parseInt(args[1]));
		conf.setInt("hidden_unit", Integer.parseInt(args[2]));
		
		String sourcePath_Train = args[3];
		String sourcePath_Test = args[4];
		String outputPath = args[5];
		int iterationNum = Integer.parseInt(args[6]);
		
		ALMain al = new ALMain(iterationNum, sourcePath_Train, sourcePath_Test, outputPath, conf);

		al.ALDriverJob();
		long endTime = System.currentTimeMillis();
		System.out.println("time:" + (endTime - startTime));

	}
	
	//mapreduce任务迭代
	public void ALDriverJob() throws IOException, ClassNotFoundException, InterruptedException {
		for (int num = 0; num < iterationNum; num++) {

			Job ALDriverJob = Job.getInstance(conf);				//创建MapReduce任务
			ALDriverJob.setJobName("ALDriverJob"+num);				//设置任务名称
			ALDriverJob.setJarByClass(AL.class);					//任务的主程序
			ALDriverJob.getConfiguration().set("trainPath", sourcePath_Train);		//设置有类别数据路径

			ALDriverJob.setMapperClass(AL.ALMapper.class);							//设置map方法
			ALDriverJob.setMapOutputKeyClass(DoubleWritable.class);					//map方法的输出key类型
			ALDriverJob.setMapOutputValueClass(Text.class);							//map方法输出value类型

			ALDriverJob.setReducerClass(AL.ALReduce.class);							//设置reudce方法
			ALDriverJob.setOutputKeyClass(NullWritable.class);						//reduce方法输出key类型
			ALDriverJob.setOutputValueClass(Text.class);							//reduce方法输出value类型

			FileInputFormat.addInputPath(ALDriverJob, new Path(sourcePath_Test));		//设置任务的输入文件路径（无类标数据）
			FileOutputFormat.setOutputPath(ALDriverJob, new Path(outputPath+"/train_"+(num+1)+"/"));   //设置任务输出文件路径（与有类别数据路径相同）

			ALDriverJob.waitForCompletion(true);										//开始执行
			
		}
		
		System.out.println("finished!");

	}

}
