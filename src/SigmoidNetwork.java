/*
 * Created by Axel_ST on 25/01/2023
 */

public class SigmoidNetwork {
    private final int BIAS = 0;
    private final int WEIGHTS = 0;
    private int numLayers;
    private int[] sizes;
    
    public SigmoidNetwork(int... sizes) {
        this.sizes = sizes;
        this.numLayers = sizes.length;
    }
    
    private int[] feedForward(int[] inputs) {
        int[] outputs = null;
        for (int i = 1; i < numLayers; i++) {
            int size = sizes[i];
            int[] z = new int [size];
            outputs = new int [size];
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < inputs.length; k++) {
                    z[j] += WEIGHTS * inputs[k];
                }
                z[j] += BIAS;
                outputs[j] = z[j] > 0 ? 1 : 0;
            }
            inputs = outputs;
        }
        return outputs;
    }
    
    public static void main(String[] args) {
        SigmoidNetwork net = new SigmoidNetwork(2, 3, 2);
        int[] inputs = {1, 0};
        int[] outputs = net.feedForward(inputs);
        
        for(int i = 0; i < outputs.length; i++) {
            System.out.println(outputs[i]);
        }
    }
}
