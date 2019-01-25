package network;

import java.util.Arrays;

public class Driver {

	public static void main(String[] args) {
		
		Network net = new Network(new int[]{3, 15, 2});
		
		double[] targets = {0, 1};
		double[] inputs = {1, 1, 0};
		
		for(int i = 0; i < 1000; ++i) {
			net.train(targets, inputs);
			//System.out.println(Arrays.toString(net.run(inputs)));
		}
		
		
		System.out.println(Arrays.toString(net.run(inputs)));

	}

}
