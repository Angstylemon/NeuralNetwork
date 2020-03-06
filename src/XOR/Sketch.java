package XOR;

import java.util.ArrayList;
import java.util.List;

import NeuralNetwork.NeuralNetwork;
import NeuralNetwork.Painter;
import processing.core.PApplet;
import processing.core.PVector;
import source.Matrix;

public class Sketch extends PApplet {
	private NeuralNetwork nn;
	private Painter p;
	
	private int width = 800;
	private int height = 800;
	
	List<double[]> training_data = new ArrayList<double[]>();
	List<double[]> targets = new ArrayList<double[]>();
	
	
	public static void main(String[] args) {
		PApplet.main("XOR.Sketch");
	}
	
	public void settings() {
		size(width, height);
	}
	
	public void setup() {
//		training_data.add(new double[] {0, 0});
//		training_data.add(new double[] {0, 0.5});
//		training_data.add(new double[] {0, 1});
//		training_data.add(new double[] {0.5, 0});
//		training_data.add(new double[] {0.5, 0.5});
//		training_data.add(new double[] {0.5, 1});
//		training_data.add(new double[] {1, 0});
//		training_data.add(new double[] {1, 0.5});
//		training_data.add(new double[] {1, 1});
//		
//		targets.add(new double[] {0});
//		targets.add(new double[] {1});
//		targets.add(new double[] {0});
//		targets.add(new double[] {1});
//		targets.add(new double[] {0});
//		targets.add(new double[] {1});
//		targets.add(new double[] {0});
//		targets.add(new double[] {1});
//		targets.add(new double[] {0});
		
		for (double i = 0; i <= 1; i += 0.25) {
			for (double j = 0; j <= 1; j += 0.25) {
				training_data.add(new double[] {i, j});
			}
		}

		targets.add(new double[] {0});
		targets.add(new double[] {0});
		targets.add(new double[] {0});
		targets.add(new double[] {1});
		targets.add(new double[] {0});
		targets.add(new double[] {0});
		targets.add(new double[] {1});
		targets.add(new double[] {0});
		targets.add(new double[] {1});
		targets.add(new double[] {1});
		targets.add(new double[] {0});
		targets.add(new double[] {0});
		targets.add(new double[] {0});
		targets.add(new double[] {0});
		targets.add(new double[] {1});
		targets.add(new double[] {0});
		targets.add(new double[] {1});
		targets.add(new double[] {0});
		targets.add(new double[] {1});
		targets.add(new double[] {1});
		targets.add(new double[] {0});
		targets.add(new double[] {0});
		targets.add(new double[] {0});
		targets.add(new double[] {1});
		targets.add(new double[] {0});
		
		nn = new NeuralNetwork(2, new int[]{20, 15}, 1);
		p = new Painter(this, nn);
	}
	
	public void draw() {
		background(255);
		
		for (int i = 0; i < 1000; i++) {
			int index = (int)(Math.random()*training_data.size());
			
			nn.train(training_data.get(index), targets.get(index));
		}
		
		
		int resolution = 10;
		double cols = (double)(width/resolution);
		double rows = (double)(height/resolution);
		
		for (int i = 0; i < cols; i++) {
			for (int j = 0; j < rows; j++) {
				double x1 = (double)i/cols;
				double x2 = (double)j/rows;
				double[] inputs = new double[] {x1, x2};
				double[] y = nn.feedForward(inputs);
				
				noStroke();
				fill((float)y[0] * 255);
				rect(i*resolution, j*resolution, resolution, resolution);
			}
		}
		
		PVector xbounds = new PVector(0, width);
		PVector ybounds = new PVector(0, height);
		
//		p.displayNetwork(xbounds, ybounds);
	}
}
