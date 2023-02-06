/*
 * Created by Axel_ST on 06/02/2023
 */

package part_3;

public class SigmoidNetwork {
    private final double BIAS = 1;
    private final double WEIGHTS = 3;
    private int numLayers;
    private int[] sizes;
    
    public SigmoidNetwork(int... sizes) {
        this.sizes = sizes;
        this.numLayers = sizes.length;
    }
    
    private double[] feedForward(double[] inputs) {
        double[] outputs = null;
        for (int i = 1; i < numLayers; i++) {
            int size = sizes[i];
            int[] z = new int[size];
            outputs = new double[size];
            for (int j = 0; j < size; j++) {
                for (double input : inputs) {
                    z[j] += WEIGHTS * input;
                }
                z[j] += BIAS;
                outputs[j] = 1 / (1 + Math.exp(z[j]));
            }
            inputs = outputs;
        }
        return outputs;
    }
    
    public static void main(String[] args) {
        SigmoidNetwork net = new SigmoidNetwork(2, 3, 2);
        double[] inputs = {1, 0};
        double[] outputs = net.feedForward(inputs);
        
        for (double output : outputs) {
            System.out.println(output);
        }
    }
}
