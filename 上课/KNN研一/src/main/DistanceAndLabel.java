/**
 * 
 */
package main;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;
/**
 * @author qjx
 *	
 */
//实现了WritableComparable接口，从而实现了hadoop的序列化
public class DistanceAndLabel implements WritableComparable<DistanceAndLabel> {

	public double distance;		//距离
	public double label;		//类别
	

	public DistanceAndLabel(double distance, double label) {
		super();
		this.distance = distance;
		this.label = label;
	}

	public DistanceAndLabel() {
		super();
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		this.distance = in.readDouble();
		this.label = in.readDouble();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeDouble(this.distance);
		out.writeDouble(label);

	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		long temp;
		temp = Double.doubleToLongBits(distance);
		result = prime * result + (int) (temp ^ (temp >>> 32));
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
		DistanceAndLabel other = (DistanceAndLabel) obj;
		if (Double.doubleToLongBits(distance) != Double.doubleToLongBits(other.distance))
			return false;
		return true;
	}

	@Override
	public int compareTo(DistanceAndLabel o) {
		if(o.equals(this)) {
			return 0;
		}else if(o.distance < this.distance) {
			return -1;
		}else {
			return 1;
		}
	}

	@Override
	public String toString() {
		return "DistanceAndLabel [distance=" + distance + ", label=" + label + "]";
	}
	

}
