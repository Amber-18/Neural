package network;

import java.util.Arrays;

public class Layer {
	
	private double[][] weights;
	private double[] biases;
	private double[] inputs;
	private double[] outputs;
	private double[] errors;
	private int numWeights;
	private int size;
	
	/** Constructs a neural network Layer of neurons
	 * @param size The number of neurons in this layer
	 * @param numWeights The number of weights each neuron will have*/
	public Layer(int size, int numWeights) {
		this.weights = new double[size][numWeights];
		this.biases = new double[size];
		this.outputs = new double[size];
		this.numWeights = numWeights;
		this.size = size;
		
		for(int neuron = 0; neuron < size; ++neuron) {
			
			this.biases[neuron] = Math.random();
			
			for(int weight = 0; weight < numWeights; ++weight) {
				this.weights[neuron][weight] = Math.random();
			}
		}
	}
	
	public double[] run(double[] inputs) {
		if(this.numWeights != inputs.length) {
			System.out.println("Inputs for this layer are not compatible with the number of weights!");
		}
		
		// initial variables
		// specific weight for every calculation
		// sum of all the weights * input values
		// array of output values for this layer that is the size of this layer
		double weight = 0;
		double sum = 0;
		double input;
		double[] outputs = new double[size];
		
		// for every neuron in this layer
		for(int neuron = 0; neuron < size; ++neuron) {
			
			// for every weight this neuron has
			for(int w = 0; w < numWeights; ++w) {
				
				// get weight and input value
				weight = this.weights[neuron][w];
				input = inputs[w];
				
				// sum all weight and input value products
				sum += weight * input;
			}
			
			sum += this.biases[neuron];
			sum = sigmoid(sum);
			
			outputs[neuron] = sum;
			sum = 0;
		}
		
		this.inputs = inputs;
		this.outputs = outputs;
		return outputs;
		
	}
	
	private static double sigmoid(double x){

        x = 1 / (1 + Math.exp((-1)*x));
        return x;

    }
	
	/** Set the double[] errors for this layer*/
	public void setErrors(double[] errors) {
		
		// TODO do the errors need to Math.abs?
		//for(int i = 0; i < errors.length; ++i) {
		//	errors[i] = Math.abs(errors[i]);
		//}
		
		this.errors = errors;
		
	}
	
	public void train(double learningRate) {
		
		// for every neuron in this layer
		for(int neuron = 0; neuron < size; ++neuron) {
					
			// for every weight this neuron has
			for(int w = 0; w < numWeights; ++w) {
				weights[neuron][w] += learningRate * errors[neuron] * inputs[w];
			}
			
			biases[neuron] += learningRate * errors[neuron];
			
		}
	}
	
	public double[][] getWeights() {
		return this.weights;
	}
	
	public double getNumWeights() {
		return this.numWeights;
	}
	
	public double getSize() {
		return this.size;
	}
	
	public double[] getOutputs() {
		return this.outputs;
	}
	
	public double[] getErrors() {
		return this.errors;
	}
}
