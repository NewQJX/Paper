package main;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import org.apache.hadoop.io.WritableComparable;


//实现了WritableComparable接口，从而实现了hadoop的序列化
public class Instance implements WritableComparable<Instance> {
	private double[] attributeValue;		//样例的属性
	private double label;					//样例的类别

	/**
	 * a line of form a1 a2 ...and lable
	 * 
	 * @param line
	 */
	public Instance(String line) {
		String[] value = line.split(" ");
		attributeValue = new double[value.length - 1];
		for (int i = 0; i < attributeValue.length; i++) {
			attributeValue[i] = Double.parseDouble(value[i]);
		}
		label = Double.parseDouble(value[value.length - 1]);
	}
	
	public Instance() {
		super();
		// TODO Auto-generated constructor stub
	}

	public Instance(double[] attribute,double label) {
		this.attributeValue = attribute;
		this.label = label;
	}

	public double[] getAtrributeValue() {
		return attributeValue;
	}

	public double getLable() {
		return label;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + Arrays.hashCode(attributeValue);
		long temp;
		temp = Double.doubleToLongBits(label);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Instance other = (Instance) obj;
		if (!Arrays.equals(attributeValue, other.attributeValue))
			return false;
		
		return true;
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		int len = in.readInt();
		this.attributeValue = new double[len];
		for (int i = 0; i < len; i++) {
			this.attributeValue[i] = in.readDouble();
		}

		this.label = in.readDouble();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(this.attributeValue.length);
		for (double a : this.attributeValue) {
			out.writeDouble(a);
		}
		out.writeDouble(label);
	}

	@Override
	public int compareTo(Instance o) {
		if (!o.equals(this)) {
			return -1;
		}
		return 0;
	}

	@Override
	public String toString() {
		return "Instance [attributeValue=" + Arrays.toString(attributeValue) + ", label=" + label + "]";
	}
	

}
