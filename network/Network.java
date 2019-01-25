package network;

import java.util.ArrayList;
import java.util.Arrays;

public class Network{
	// class creates a Network object with variables below
	
	/** The learning rate of the network to be used during training*/
	private double learningRate = 0.2;

	/** number of layers in the network */
    private int numLayers;

    /** sizes of each layer in the network organized in an array */
    private int[] layerSizes;
    
    /** ArrayList of the Layers objects */
    private ArrayList<Layer> layers;
    
    public Network(int[] layerSizes){
        this.layerSizes = layerSizes;
        this.numLayers = layerSizes.length;
        this.layers = new ArrayList<>();
        
        for(int i = 1; i < this.layerSizes.length; ++i) {
        	this.layers.add(new Layer(layerSizes[i], layerSizes[i-1]));
        }
        
    }

    public double[] run(double[] inputs){
        
    	double[] outputs = {};
    	
    	for(int i = 0; i < layers.size(); ++i) {
    		outputs = layers.get(i).run(inputs);
    		
    		inputs = outputs;
    	}
    	
        return outputs;
    }
    
    public void train(double[] targets, double[] inputs) {
    	double[] outputs = this.run(inputs);
    	double[] output_errors = new double[outputs.length];
    	
    	// get array of errors from errors of output neurons
		// output_error = (y - a) * (a * (1-a) )
		// a is actual output, y is target/expected
    	for(int i = 0; i < outputs.length; ++i) {
    		output_errors[i] = (targets[i] - outputs[i]) * (outputs[i]) * (1 - outputs[i]);
    	}
    	
    	// set output_errors in output layer
    	this.layers.get(layers.size()-1).setErrors(output_errors);
    	
    	double[][] weights;
    	double numNeurons;
    	double numForwardNeurons;
    	double[] hidden_errors = null;
    	double[] forward_errors;
    	double[] hidden_outputs;
    	double sum;
    	
    	// for every hidden layer, starting with the last hidden layer, set errors for each layer
    	// errors, once set, will be used in training
    	
    	// for every layer in the network
    	for(int layer = layers.size() - 1; layer > 0; --layer) {
    		
    		// get the weights of the layer ahead of the one we are calculating the errors of
    		weights = layers.get(layer).getWeights();
    		
    		// number of neurons in the forward layer and layer we are calculating errors for
    		numForwardNeurons = layers.get(layer).getSize();
    		numNeurons = layers.get(layer-1).getSize();
    		hidden_errors = new double[(int)numNeurons];
    		
    		// array of the errors of the foward layer
    		forward_errors = layers.get(layer).getErrors();
    		
    		// get the outputs of the layer we are calculating the errors for
    		// if this the first hidden layer, index is 0, outputs are the initial array of inputs
    		try {
    			hidden_outputs = layers.get(layer - 1).getOutputs();
    		} catch (ArrayIndexOutOfBoundsException e) {
    			hidden_outputs = inputs;
    		}
    		
    		
    		
    		// for every neuron in the layer we are getting errors for
    		for(int neuron = 0; neuron < (int)numNeurons; ++neuron) {
    			
    			sum  = 0;
    			
    			// for every neuron in the forward layer
    			for(int f_neuron = 0; f_neuron < numForwardNeurons; ++f_neuron) {
    				
    				// look up error of a hidden neuron to understand equation below
    				sum += weights[f_neuron][neuron] 
    						* forward_errors[f_neuron] 
    						* hidden_outputs[neuron] 
    						* (1 - hidden_outputs[neuron]);
    			}
    			
    			hidden_errors[neuron] = sum;
    		}
    		
    		layers.get(layer-1).setErrors(hidden_errors);
    	}
    	
    	
    	// errors are now calculated !!
    	// each layer object should now have a double[] of errors that are correct! 
    	
    	// for every layer !
    	for(int layer = 0; layer < layers.size(); ++ layer) {
    		layers.get(layer).train(learningRate);
    	}
    	
    	
    	
    }
    
    
}
