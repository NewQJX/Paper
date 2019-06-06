package al;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import elm.elm;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.NotConvergedException;

public class AL {
	private static int NUM_CLASS;	//数据类别
	private static int NUM_SELECT; // 每次迭代挑选重要样例数量
	
	
	/**
	 *    输入有类别信息数据， 输出<熵值， 数据信息>
	 */
	public static class ALMapper extends Mapper<LongWritable, Text, DoubleWritable, Text> {
		private DenseMatrix trainfile_matrix;											//有类标数据矩阵
		private ArrayList<String> arraylist = new ArrayList<String>();
		private int m;																	//有类标数据数量
		private int n;																	//数据维度
		private int hidden_unit;														//elm隐层节点数量
		private elm e;			
		
		
		
		/*
		 *   用有类别信息的数据训练elm分类器，并获取分类 器精度
		 */
		@Override
		protected void setup(Context context) throws IOException, InterruptedException {
			// initialize
			NUM_SELECT = context.getConfiguration().getInt("NUM_SELECT", 10);
			NUM_CLASS = context.getConfiguration().getInt("NUM_CLASS", 2);
			hidden_unit = context.getConfiguration().getInt("hidden_unit", 20);
			
			//读取有类别信息的数据
			FileSystem fs = FileSystem.get(context.getConfiguration());
			FileStatus[] fileList = fs.listStatus(new Path(context.getConfiguration().get("trainPath")));
			BufferedReader in = null;
			FSDataInputStream fsi = null;
			String line = null;
			for (int i = 0; i < fileList.length; i++) {
				if (!fileList[i].isDirectory()) {
					fsi = fs.open(fileList[i].getPath());
					in = new BufferedReader(new InputStreamReader(fsi, "UTF-8"));
					while ((line = in.readLine()) != null) {
						// System.out.println("read a line:" + line);
						arraylist.add(line);
						String[] strs_t = line.split(" ");
						n = strs_t.length;
					}
				} else {
					FileStatus[] filelist_in = fs.listStatus(fileList[i].getPath());
					for (int j = 0; j < filelist_in.length; j++) {
						if (!filelist_in[j].isDir()) {
							fsi = fs.open(filelist_in[j].getPath());
							in = new BufferedReader(new InputStreamReader(fsi, "UTF-8"));
							while ((line = in.readLine()) != null) {
								// System.out.println("read a line:" + line);
								arraylist.add(line);
								String[] strs_t = line.split(" ");
								n = strs_t.length;
							}
						}
					}
				}
			}
			m = arraylist.size();
			Counter numberConuter = context.getCounter("samples_number", m + "");		//有类别数据数量
			
			trainfile_matrix = new DenseMatrix(m, n);				
			for (int mm = 0; mm < arraylist.size(); mm++) {
				String str_t = arraylist.get(mm);
				String[] str_t_s = str_t.split(" ");
				for (int nn = 0; nn < str_t_s.length; nn++) {
					trainfile_matrix.set(mm, nn, Double.parseDouble(str_t_s[nn]));
				}
			}

			e = new elm(1, hidden_unit, "sig");											//创建elm分类器
			try {
				e.train(trainfile_matrix, NUM_CLASS);									//训练
			} catch (NotConvergedException e1) {
				e1.printStackTrace();
			}
			Counter accuracyCounter = context.getCounter("accuracy", e.getTrainingAccuracy() + "");			//获取分类器精度

		}
		
		/*
		 *	按行读取无类别信息的数据， 并计算其熵值。
		 *	key: 行偏移量
		 *  value: 数据信息
		 */
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String line = value.toString();
			String[] lines = line.split(" ");
			DenseMatrix test_matrix = new DenseMatrix(1, lines.length);
			for (int i = 0; i < lines.length; i++) {
				test_matrix.set(0, i, Double.parseDouble(lines[i]));
			}
			
			e.test(test_matrix);											
			DenseMatrix d = e.getOutMatrix();										//得到属于每一类别的概率
			double sum = 0;
			for (int i = 0; i < d.numColumns(); i++) {
				sum += Math.exp(d.get(0, i));
			}
			for (int i = 0; i < d.numColumns(); i++) {
				d.set(0, i, Math.exp(d.get(0, i)) / sum);
			}
			double H = 0;
			for (int j = 0; j < d.numColumns(); j++) {
				H += Math.log(d.get(0, j)) / Math.log(2.0) * d.get(0, j);
			}
			context.write(new DoubleWritable(-1 / H), new Text(value));				//输出<熵值， 数据信息>
		}
	}
	
	/**
	 *    输入：<熵值， 有类别信息的数据>  已按熵值排序
	 *    输出：熵值最大的前NUM_SELECT个数据
	 */
	public static class ALReduce extends Reducer<DoubleWritable, Text, NullWritable, Text> {
		@Override
		protected void setup(Context context) throws IOException, InterruptedException {
			NUM_SELECT = context.getConfiguration().getInt("NUM_SELECT", 10);
			NUM_CLASS = context.getConfiguration().getInt("NUM_CLASS", 2);
		}

		public void reduce(DoubleWritable key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			for (Text text : values) {
				if (NUM_SELECT-- > 0) {
					context.write(NullWritable.get(), new Text(text));
				}
			}

		}
	}
}
