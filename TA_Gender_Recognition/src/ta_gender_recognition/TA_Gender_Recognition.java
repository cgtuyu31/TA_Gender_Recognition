/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ta_gender_recognition;

import java.util.ArrayList;

/**
 *
 * @author Tuyu
 */
public class TA_Gender_Recognition {

    /**
     * @param args the command line arguments
     */
    @SuppressWarnings("empty-statement")
    public static void main(String[] args) {
        // TODO code application logic here
        ArrayList<double[]> x = new ArrayList<>();
        double[] tmp = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        x.add(tmp);
        double[] tmp1 = {11, 12, 13, 14, 15, 16, 17, 18, 19};
        x.add(tmp1);
        double[] tmp2 = {21, 22, 23, 24, 25, 26, 27, 28, 29};
        x.add(tmp2);

        ArrayList<double[]> y = new ArrayList<>();
        for (double[] tes : x) {
            y.add(getChunkArray(tes, 3, 0));
        }

        for (double[] tes : y) {
            for (int i = 0; i < tes.length; i++) {
                System.out.print(tes[i] + " ");
            }
            System.out.println("");
        }

        System.out.println("-----------------------");

        ArrayList<double[]> z = new ArrayList<>();
        for (double[] tes : x) {
            z.add(getChunkArray(tes, 3, 1));
        }

        for (double[] tes : z) {
            for (int i = 0; i < tes.length; i++) {
                System.out.print(tes[i] + " ");
            }
            System.out.println("");
        }

        System.out.println("-----------------------");

//        ArrayList<double[]> a = new ArrayList<>();
//        for (int i = 0; i < x.size(); i++) {
//            double result[] = new double[y.get(i).length + z.get(i).length];
//            for (int j = 0; j < 2; j++) {
//                System.arraycopy(y.get(i), 0, result, 0, y.get(i).length);
//                System.arraycopy(z.get(i), 0, result, y.get(i).length, z.get(i).length);
//            }
//            a.add(result);
//        }
//        for (double[] tes : a) {
//            for (int i = 0; i < tes.length; i++) {
//                System.out.print(tes[i] + " ");
//            }
//            System.out.println("");
//        }
    }

    public static double[] getChunkArray(double[] array, int chunkSize, int block) {
        int numOfChunks = (int) Math.ceil((double) array.length / chunkSize);
        double[][] output = new double[numOfChunks][];

        for (int i = 0; i < numOfChunks; ++i) {
            int start = i * chunkSize;
            int length = Math.min(array.length - start, chunkSize);

            double[] temp = new double[length];
            System.arraycopy(array, start, temp, 0, length);
            output[i] = temp;
        }

        return output[block];
    }

    // Example usage:
    //
    // int[] numbers = {1, 2, 3, 4, 5, 6, 7};
    // int[][] chunks = chunkArray(numbers, 3);
    //
    // chunks now contains [
    //                         [1, 2, 3],
    //                         [4, 5, 6],
    //                         [7]
    //                     ]
    public static int[][] chunkArray(int[] array, int chunkSize) {
        int numOfChunks = (int) Math.ceil((double) array.length / chunkSize);
        int[][] output = new int[numOfChunks][];

        for (int i = 0; i < numOfChunks; ++i) {
            int start = i * chunkSize;
            int length = Math.min(array.length - start, chunkSize);

            int[] temp = new int[length];
            System.arraycopy(array, start, temp, 0, length);
            output[i] = temp;
        }

        return output;
    }
}
