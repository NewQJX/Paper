package util;

import java.util.ArrayList;

import main.DistanceAndLabel;

public class Sort {
	
	public static ArrayList<DistanceAndLabel> getNearest(ArrayList<DistanceAndLabel> list ,int k){
		ArrayList<DistanceAndLabel> result = new ArrayList<>();
		
		for(int i = 0; i< list.size(); i++) {
			if(result.size() < k) {
				result.add(list.get(i));
			}else {
				int index = indexOfMax(result);
				if(list.get(i).distance < result.get(index).distance) {
					result.remove(index);
					result.add(list.get(i));
				}
			}
		}
		
		return result;
	}
	public static int indexOfMax(ArrayList<DistanceAndLabel> array){
		int index = -1;
		Double min = Double.MIN_VALUE; 
		for (int i = 0;i < array.size();i++){
			if(array.get(i).distance > min){
				min = array.get(i).distance;
				index = i;
			}
		}
		return index;
	}
}
