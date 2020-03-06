package NeuralNetwork;

import source.Matrix;

public class NeuralNetwork {
	private int _inputNodes;
	private int[] _hiddenNodes;
	private int _outputNodes;
	
	private Matrix _weightsIH;
	private Matrix[] _weightsHH;
	private Matrix _weightsHO;
	
	private Matrix[] _biasH;
	private Matrix _biasO;
	
	private double _learningRate = 0.1;
		
	public NeuralNetwork(int inputs, int[] hidden, int outputs) {
		//Assign node field values
		_inputNodes = inputs;
		_hiddenNodes = hidden;
		_outputNodes = outputs;
		
		//Create weight matrix from input nodes -> first layer hidden nodes
		_weightsIH = new Matrix(_hiddenNodes[0], _inputNodes);
		_weightsIH.randomise();
		
		//Create weight matrices between each layer of hidden nodes
		_weightsHH = new Matrix[_hiddenNodes.length-1];
		for (int i = 0; i < _weightsHH.length; i++) {
			_weightsHH[i] = new Matrix(_hiddenNodes[i+1], _hiddenNodes[i]);
			_weightsHH[i].randomise();
		}
		
		//Create weight matrix from last layer hidden nodes -> output nodes
		_weightsHO = new Matrix(_outputNodes, _hiddenNodes[_hiddenNodes.length-1]);
		_weightsHO.randomise();
		
		//Create bias matrices for each layer of hidden nodes
		_biasH = new Matrix[_hiddenNodes.length];
		for (int i = 0; i < _hiddenNodes.length; i++) {
			_biasH[i] = new Matrix(_hiddenNodes[i], 1);
			_biasH[i].randomise();
		}
		//Create bias matrix for output nodes
		_biasO = new Matrix(_outputNodes, 1);
		_biasO.randomise();
	}
	
	public NeuralNetwork(int inputs, int[] hidden, int outputs, Matrix ih, Matrix[] hh, Matrix ho, Matrix[] bh, Matrix bo) {
		_inputNodes = inputs;
		_hiddenNodes = hidden;
		_outputNodes = outputs;
		
		_weightsIH = ih;
		_weightsHH = hh;
		_weightsHO = ho;
		
		_biasH = bh;
		_biasO = bo;
	}
	
	public double[] feedForward(double[] input_array) {
		//Convert input array to matrix
		Matrix inputs = Matrix.convertToMatrix(input_array);
		
		//Calculate values of first layer hidden nodes
		Matrix hidden = Matrix.dotProduct(_weightsIH, inputs);
		hidden = Matrix.add(hidden, _biasH[0]);
		hidden = hidden.sigmoid();
		
		//Go through hidden node layers
		for (int i = 0; i < _weightsHH.length; i++) {
			hidden = Matrix.dotProduct(_weightsHH[i], hidden);
			hidden = Matrix.add(hidden, _biasH[i+1]);
			hidden = hidden.sigmoid();
		}
		
		//Calculate values of output nodes
		Matrix outputs = Matrix.dotProduct(_weightsHO, hidden);
		outputs = Matrix.add(outputs, _biasO);
		outputs = outputs.sigmoid();
		
//		System.out.println(output);
		
		return outputs.convertToArray();
	}
	
	public void train(double[] input_array, double[] target_array) {
		//Convert input array to matrix
		Matrix inputs = Matrix.convertToMatrix(input_array);
		
		//Array to store values of hidden layer nodes
		Matrix[] hiddenLayers = new Matrix[_hiddenNodes.length];
		
		//Calculate values of first layer hidden nodes
		hiddenLayers[0] = Matrix.dotProduct(_weightsIH, inputs);
		hiddenLayers[0] = Matrix.add(hiddenLayers[0], _biasH[0]);
		hiddenLayers[0] = hiddenLayers[0].sigmoid();
		
		//Go through hidden layers
		for (int i = 0; i < _weightsHH.length; i++) {
			hiddenLayers[i+1] = Matrix.dotProduct(_weightsHH[i], hiddenLayers[i]);
			hiddenLayers[i+1] = Matrix.add(hiddenLayers[i+1], _biasH[i+1]);
			hiddenLayers[i+1] = hiddenLayers[i+1].sigmoid();
		}
		
		//Calculate values of output nodes
		Matrix outputs = Matrix.dotProduct(_weightsHO, hiddenLayers[hiddenLayers.length-1]);
		outputs = Matrix.add(outputs, _biasO);
		outputs = outputs.sigmoid();
		
		
		
		//Error = target - output
		Matrix targets = Matrix.convertToMatrix(target_array);
		Matrix errors = Matrix.subtract(targets, outputs);
		
		//Gradient = output * (1 - output)
		//Adjust to error and learning rate
		Matrix gradients = outputs.dsigmoid();
		gradients = Matrix.multiply(gradients, errors);
		gradients.multiply(_learningRate);
		
		//Change in weight values proportional to node activation
		Matrix nodesTranspose = Matrix.transpose(hiddenLayers[hiddenLayers.length-1]);
		Matrix weightDeltas = Matrix.dotProduct(gradients, nodesTranspose);
		
		//Adjust weights and bias
		_weightsHO = Matrix.add(_weightsHO, weightDeltas);
		_biasO = Matrix.add(_biasO, gradients);
		
		//Error based on previous layer
		Matrix weightsTranspose = Matrix.transpose(_weightsHO);
		errors = Matrix.dotProduct(weightsTranspose, errors);
		//Gradients
		gradients = hiddenLayers[hiddenLayers.length-1].dsigmoid();
		gradients = Matrix.multiply(gradients, errors);
		gradients.multiply(_learningRate);
		
		for (int i = _weightsHH.length-1; i >= 0; i--) {
			//Change of weights
			nodesTranspose = Matrix.transpose(hiddenLayers[i]);
			weightDeltas = Matrix.dotProduct(gradients, nodesTranspose);
			
			//Adjust weights and bias
			_weightsHH[i] = Matrix.add(_weightsHH[i], weightDeltas);
			_biasH[i+1] = Matrix.add(_biasH[i+1], gradients);
			
			
			//Error based on previous layer
			weightsTranspose = Matrix.transpose(_weightsHH[i]);
			errors = Matrix.dotProduct(weightsTranspose, errors);
			//Gradients
			gradients = hiddenLayers[i].dsigmoid();
			gradients = Matrix.multiply(gradients, errors);
			gradients.multiply(_learningRate);
		}
		
		//Change of weights
		nodesTranspose = Matrix.transpose(inputs);
		weightDeltas = Matrix.dotProduct(gradients, nodesTranspose);
		//Adjust values
		_weightsIH = Matrix.add(_weightsIH, weightDeltas);
		_biasH[0] = Matrix.add(_biasH[0], gradients);
	}
	
	public void mutate(double mutation_rate) {
		_weightsIH.mutate(mutation_rate);
		for (Matrix m : _weightsHH) {
			m.mutate(mutation_rate);
		}
		_weightsHO.mutate(mutation_rate);
		
		for (Matrix b : _biasH) {
			b.mutate(mutation_rate);
		}
		_biasO.mutate(mutation_rate);
	}
	
	public NeuralNetwork copy() {
		Matrix _weightsIH_copy = _weightsIH.copy();
		Matrix[] _weightsHH_copy = new Matrix[_hiddenNodes.length-1];
		for (int i = 0; i < _weightsHH_copy.length; i++) {
			_weightsHH_copy[i] = _weightsHH[i].copy();
		}
		Matrix _weightsHO_copy = _weightsHO.copy();
		
		Matrix[] _biasH_copy = new Matrix[_hiddenNodes.length];
		for (int i = 0; i < _biasH_copy.length; i++) {
			_biasH_copy[i] = _biasH[i].copy();
		}
		Matrix _biasO_copy = _biasO.copy();
		
		NeuralNetwork copyNN = new NeuralNetwork(_inputNodes, _hiddenNodes, _outputNodes, _weightsIH_copy, _weightsHH_copy, _weightsHO_copy, _biasH_copy, _biasO_copy);
		return copyNN;
	}
	
	
	
	public int inputNodes() {
		return _inputNodes;
	}
	
	public int[] hiddenNodes() {
		return _hiddenNodes;
	}
	
	public int outputNodes() {
		return _outputNodes;
	}
	
	public Matrix weightsIH() {
		return _weightsIH.copy();
	}
	
	public Matrix[] weightsHH() {
		Matrix[] _weightsHH_copy = new Matrix[_hiddenNodes.length-1];
		for (int i = 0; i < _weightsHH_copy.length; i++) {
			_weightsHH_copy[i] = _weightsHH[i].copy();
		}
		
		return _weightsHH_copy;
	}
	
	public Matrix weightsHO() {
		return _weightsHO.copy();
	}
}
