/*
 * Created by Axel_ST on 13/02/2023
 */

package part_4;

import org.jblas.DoubleMatrix;

public class SigmoidNetwork {
    private int numLayers;
    private int[] sizes;
    
    private DoubleMatrix[] weights;
    private DoubleMatrix[] biases;
    
    public SigmoidNetwork(int... sizes) { //constructor
        this.sizes = sizes;
        this.numLayers = sizes.length;
        
        this.weights = new DoubleMatrix[sizes.length - 1];
        this.biases = new DoubleMatrix[sizes.length - 1];
        
        // storing weights
        final double WEIGHT = 31; // any positive value, more when bias
        double[][] customWeights = new double[][] { // manually filled weights
                {0, 0, 0, 0, 0, 0, 0, 0, WEIGHT, WEIGHT},
                {0, 0, 0, 0, WEIGHT, WEIGHT, WEIGHT, WEIGHT, 0, 0},
                {0, 0, WEIGHT, WEIGHT, 0, 0, WEIGHT, WEIGHT, 0, 0},
                {0, WEIGHT, 0, WEIGHT, 0, WEIGHT, 0, WEIGHT, 0, WEIGHT}
        };
        weights[0] = new DoubleMatrix(customWeights); // put them in a matrix view
        
        // storing biases
        for (int i = 1; i < sizes.length; i++) { // checking each layer except input layer
            double[][] temp = new double[sizes[i]][];
            for (int j = 0; j < sizes[i]; j++) { // each neuron in layer
                double[] b = new double[] {-30}; // constant value for checking; any negative value
                temp[j] = b; // storage of neuron's biases
            }
            biases[i - 1] = new DoubleMatrix(temp); // put biases into matrix
        }
    }
    
    private DoubleMatrix feedForward(DoubleMatrix a) { // new method that work with matrices
        for (int i = 0; i < numLayers - 1; i++) {
            double[] z = new double[weights[i].rows];
            for (int j = 0; j < weights[i].rows; j++) {
                z[j] = weights[i].getRow(j).dot(a) + biases[i].get(j); // z = w * x + b
            }
            DoubleMatrix output = new DoubleMatrix(z);
            a = sigmoid(output);
        }
        return a;
    }
    
    private DoubleMatrix sigmoid(DoubleMatrix z) { // separate sigmiod activation function that works with matrices
        double[] output = new double[z.length];
        for (int i = 0; i < output.length; i++) {
            output[i] = 1 / (1 + Math.exp(-z.get(i)));
        }
        return new DoubleMatrix(output);
    }
    
    public static void main(String[] args) {
        SigmoidNetwork net = new SigmoidNetwork(10, 4);
        double[] inputs = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
        DoubleMatrix outputs = net.feedForward(new DoubleMatrix(inputs));
        
        System.out.println(outputs.toString("%.0f"));
    }
}
