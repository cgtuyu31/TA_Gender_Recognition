/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Methods;

import java.util.ArrayList;


/**
 *
 * @author Tuyu
 */
public class Normalization {
    
    public static double[][] AListToArray(ArrayList<double[]> data){
        double[][] tmp = new double[data.get(0).length][data.size()];
        System.out.println("tmp[][] -> " + tmp.length + " - " + tmp[0].length);
        for (int i = 0; i < data.get(0).length; i++) {
            for (int j = 0; j < data.size(); j++) {
                tmp[i][j] = data.get(j)[i];
            }
        }
        return tmp;
    }
    
    public static double[] getChunkArray(double[] array, int chunkSize, int block) {
        int numOfChunks = (int)Math.ceil((double)array.length / chunkSize);
        double[][] output = new double[numOfChunks][];

        for(int i = 0; i < numOfChunks; ++i) {
            int start = i * chunkSize;
            int length = Math.min(array.length - start, chunkSize);

            double[] temp = new double[length];
            System.arraycopy(array, start, temp, 0, length);
            output[i] = temp;
        }

        return output[block];
    }
}
