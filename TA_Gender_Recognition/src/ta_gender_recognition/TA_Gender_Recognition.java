/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ta_gender_recognition;

import Methods.Normalization;
import Methods.PCA;
import java.util.ArrayList;
import static ta_gender_recognition.Test.getTrainTestDataFromCSV;

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
        ArrayList<String[]> listData = new ArrayList<>();
//        for (int i = 0; i < classGender.length; i++) {
//            getTrainTestData(i);
        int block = 3;

        for (int j = 0; j < block; j++) {
            System.out.println("======================================================");
            System.out.println("Block : " + (j + 1));
            ArrayList<double[]> tmp = new ArrayList<>();

            ArrayList<String[]> x = new ArrayList<>();
            String[] x1 = {"1", "2", "3"};
            x.add(x1);
            String[] x2 = {"11", "12", "13"};
            x.add(x2);
            String[] x3 = {"21", "22", "23"};
            x.add(x3);

            if (j == 0) {
                listData.addAll(x);
            } else {
                for (int k = 0; k < x.size(); k++) {
                    String[] tmp1 = listData.get(k);
                    String[] result = new String[block * 3];
                    System.arraycopy(tmp1, 0, result, 0, tmp1.length);
                    System.arraycopy(x.get(k), 0, result, j * 3, x.get(k).length);
                    listData.set(k, result);
                }
            }
        }
        System.out.println("-------------------------------------------------------------------");
        for (String[] tes : listData) {
            for (int i = 0; i < tes.length; i++) {
                System.out.print(tes[i] + " ");
            }
            System.out.println("");
        }
    }

    public static void cek() {
//        ArrayList<double[]> x = new ArrayList<>();
//        double[] tmp = {1, 2, 3, 4, 5, 6, 7, 8, 9};
//        x.add(tmp);
//        double[] tmp1 = {11, 12, 13, 14, 15, 16, 17, 18, 19};
//        x.add(tmp1);
//        double[] tmp2 = {21, 22, 23, 24, 25, 26, 27, 28, 29};
//        x.add(tmp2);
//
//        ArrayList<double[]> y = new ArrayList<>();
//        for (double[] tes : x) {
//            y.add(getChunkArray(tes, 3, 0));
//        }
//
//        for (double[] tes : y) {
//            for (int i = 0; i < tes.length; i++) {
//                System.out.print(tes[i] + " ");
//            }
//            System.out.println("");
//        }
//
//        System.out.println("-----------------------");
//
//        ArrayList<double[]> z = new ArrayList<>();
//        for (double[] tes : x) {
//            z.add(getChunkArray(tes, 3, 1));
//        }
//
//        for (double[] tes : z) {
//            for (int i = 0; i < tes.length; i++) {
//                System.out.print(tes[i] + " ");
//            }
//            System.out.println("");
//        }
//
//        System.out.println("-----------------------");

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

    public static String[] getChunkArray(double[] array, int chunkSize, int block) {
        int numOfChunks = (int) Math.ceil((double) array.length / chunkSize);
        String[][] output = new String[numOfChunks][];

        for (int i = 0; i < numOfChunks; ++i) {
            int start = i * chunkSize;
            int length = Math.min(array.length - start, chunkSize);

            String[] temp = new String[length];
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
