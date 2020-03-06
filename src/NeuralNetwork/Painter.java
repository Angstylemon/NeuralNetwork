package NeuralNetwork;

import processing.core.PApplet;
import processing.core.PVector;
import source.Matrix;

public class Painter {
	private PApplet parent;
	private NeuralNetwork nn;
	
	int[] posCon = {0, 255, 0};
	int[] negCon = {255, 0, 0};
	
	public Painter(PApplet p, NeuralNetwork n) {
		parent = p;
		nn = n;
	}
	
	public void displayNetwork(PVector xBounds, PVector yBounds) {
		parent.fill(255);
		parent.stroke(0);
		
		int inputNodes = nn.inputNodes();		
		int[] hiddenNodes = nn.hiddenNodes();
		int outputNodes = nn.outputNodes();
		
		//Get maximum number of nodes in any hidden layer
		int hiddenNodesMax = hiddenNodes[0];
		for (int i = 1; i < hiddenNodes.length; i++) {
			if (hiddenNodes[i] > hiddenNodesMax) {
				hiddenNodesMax = hiddenNodes[i];
			}
		}
		//Maximum number number of nodes in any layer
		int maxNodes = Math.max(Math.max(inputNodes, outputNodes), hiddenNodesMax);		
		
		//Calculate what size of each circle should be
		float nodeSize;
		//Node size based on horizontal constraints and number of layers
		float xNodeSize = (xBounds.y - xBounds.x)/(3*(2 + hiddenNodes.length) - 1);
		//Node size based on vertical constraints and maximum number of nodes in layers
		float yNodeSize = (yBounds.y - yBounds.x)/(3*maxNodes - 1);
		//Circle size should be minimum of these to ensure no overlap
		nodeSize = Math.min(xNodeSize, yNodeSize);
		
		
		
		//Calculate draw points for all circles
		//Setup storage for draw points
		PVector[] inputCircles = new PVector[inputNodes];
		PVector[][] hiddenCircles = new PVector[hiddenNodes.length][];
		for (int i = 0; i < hiddenNodes.length; i++) {
			hiddenCircles[i] = new PVector[hiddenNodes[i]];
		}
		PVector[] output_circles = new PVector[outputNodes];
		
		//Get y range
		float yRange = yBounds.y - yBounds.x;
		
		//Draw input node circles
		float stepSize = yRange/inputNodes;
		for (int i = 0; i < inputNodes; i++) {
			float x_pos = xBounds.x + 20 + nodeSize/2;
			float y_pos = (float) (yBounds.x + stepSize * (i + 0.5));
			
			inputCircles[i] = new PVector(x_pos, y_pos);
			parent.circle(x_pos, y_pos, nodeSize);
		}
		
		//Draw hidden node circles
		//Get bounds and ranges
		float xStart = xBounds.x + 2*nodeSize;
		float xEnd = xBounds.y - 2*nodeSize;
		float xRange = xEnd - xStart;
		float xStep = xRange/hiddenNodes.length;
		
		for (int h = 0; h < hiddenNodes.length; h++) {
			stepSize = yRange/hiddenNodes[h];
			
			for (int i = 0; i < hiddenNodes[h]; i++) {
				float xPos = (float) (xStart + h*xStep + 0.5*xStep);
				float yPos = (float) (yBounds.x + i*stepSize + 0.5*stepSize);
				
				hiddenCircles[h][i] = new PVector(xPos, yPos);
				parent.circle(xPos,  yPos, nodeSize);
			}
		}
		
		//Draw output nodes
		stepSize = yRange/outputNodes;
		for (int i = 0; i < outputNodes; i++) {
			float xPos = xBounds.y - 20 - nodeSize/2;
			float yPos = (float) (yBounds.x + i*stepSize + 0.5*stepSize);
			
			output_circles[i] = new PVector(xPos, yPos);
			parent.circle(xPos,  yPos,  nodeSize);
		}
		
		
		
		//Draw everything
		
		//Draw input-hidden connections
		Matrix weightsIH = nn.weightsIH();
		for (int i = 0; i < inputNodes; i++) {
			for (int j = 0; j < hiddenNodes[0]; j++) {
				float weight = (float) weightsIH.element(j, i);
				
				if (weight > 0) {
					parent.stroke(posCon[0], posCon[1], posCon[2], 255*weight);
				} else {
					parent.stroke(negCon[0], negCon[1], negCon[2], 255*-weight);
				}
				
				parent.line(inputCircles[i].x, inputCircles[i].y, hiddenCircles[0][j].x, hiddenCircles[0][j].y);
			}
		}
		
		//Draw hidden-hidden connections
		Matrix[] listWeightsHH = nn.weightsHH();
		for (int h = 0; h < listWeightsHH.length; h++) {
			Matrix weightsHH = listWeightsHH[h];
			for (int i = 0; i < hiddenNodes[h]; i++) {
				for (int j = 0; j < hiddenNodes[h+1]; j++) {
					float weight = (float) weightsHH.element(j, i);
					
					if (weight > 0) {
						parent.stroke(posCon[0], posCon[1], posCon[2], 255*weight);
					} else {
						parent.stroke(negCon[0], negCon[1], negCon[2], 255*-weight);
					}
					
					parent.line(hiddenCircles[h][i].x, hiddenCircles[h][i].y, hiddenCircles[h+1][j].x, hiddenCircles[h+1][j].y);
				}
			}
		}
		
		//Draw hidden-output connections
		Matrix weightsHO = nn.weightsHO();
		for (int i = 0; i < hiddenNodes[hiddenNodes.length-1]; i++) {
			for (int j = 0; j < outputNodes; j++) {
				float weight = (float) weightsHO.element(j, i);
				
				if (weight > 0) {
					parent.stroke(posCon[0], posCon[1], posCon[2], 255*weight);
				} else {
					parent.stroke(negCon[0], negCon[1], negCon[2], 255*-weight);
				}
				
				parent.line(hiddenCircles[hiddenNodes.length-1][i].x, hiddenCircles[hiddenNodes.length-1][i].y, output_circles[j].x, output_circles[j].y);
			}
		}
	}
	
	public void setNetwork(NeuralNetwork n) {
		nn = n;
	}
}
