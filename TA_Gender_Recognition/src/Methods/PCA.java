package Methods;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import static Methods.CsvUtils.getDataFromCSV;
import static Methods.CsvUtils.getDataFromText2D;
import static Methods.CsvUtils.writeToCSV;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import org.opencv.core.Core;
import ta_gender_recognition.GUI;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author Tuyu
 */
public class PCA {

    static int K = 1240;
    static ArrayList<double[]> listData;
    static ArrayList<double[]> listNorm;
    static ArrayList<String[]> listWeight;
    static double[] avg;
    static int nFeature = 7936;
    static Matrix matrixB;
    static Matrix matrixCovariance; // [M x nFeature] * [nFeature x M] = M x M
    static Matrix eigenVector; // M x M
    static Matrix eigenFace; // [nFeature x M] * [M x M] = nFeature x M
    static String gender;
    static final String HEADER_PRINT = GUI.PATH_HEADER_TRAINING;

    public PCA(int K, int nFeature) {
        this.K = K;
        this.nFeature = nFeature;
    }
    
    ///test
    public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException, IOException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        cobaTest();
    }

    public static void train(ArrayList<double[]> listHist, String gen) throws FileNotFoundException, UnsupportedEncodingException {
        nFeature = listHist.get(0).length;
        listData = listHist;
        avg = new double[nFeature];
        gender = gen;
        
        //AVERAGE
        calcAvg();
        writeToCSV(avg, HEADER_PRINT + "avg_" + gender + ".csv");

        //NORMALIZE
        normalize();
        writeToCSV(listNorm, HEADER_PRINT + "listNorm_" + gender + ".csv");

        //COVARIANCE MATRIX
        calcCovarianceMatrix();
        writeToCSV(matrixB, HEADER_PRINT + "matrixB_" + gender + ".csv");
        writeToCSV(matrixB.transpose(), HEADER_PRINT + "matrixBtranspose_" + gender + ".csv");
        writeToCSV(matrixCovariance, HEADER_PRINT + "matrixCovariance_" + gender + ".csv");

        //EIGEN VALUE & EIGEN VECTOR
        calcEigenValueAndVector();
        writeToCSV(eigenVector, HEADER_PRINT + "EigenVec_" + gender + ".csv");

        //EIGEN FACE
        calcEigenface();
        writeToCSV(eigenFace, HEADER_PRINT + "EigenFace_" + gender + ".csv");

        //WEIGHT
        calcWeight();
        System.out.println("Training Done!");
    }

    public static void calcAvg() throws FileNotFoundException, UnsupportedEncodingException {
        System.out.println("nFeature = "+nFeature);
        System.out.println("listData.size() = "+listData.size());
        for (int i = 0; i < nFeature; i++) {
            for (int j = 0; j < listData.size(); j++) {
                avg[i] += listData.get(j)[i];
            }
            avg[i] = avg[i] / listData.size();
        }
    }

    public static void normalize() {
        double[] tmp;
        listNorm = new ArrayList<>();
        for (int i = 0; i < listData.size(); i++) {
            tmp = new double[nFeature];
            for (int j = 0; j < nFeature; j++) {
                tmp[j] = listData.get(i)[j] - avg[j];
            }
            listNorm.add(tmp);
        }
//        for (int i = 0; i < listNorm.size(); i++) {
//            for (int k = 0; k < listNorm.get(i)[k]; k++) {
//                System.out.print(" " + listNorm.get(i));
//            }
//            System.out.println("");
//        }
    }

    public static void calcCovarianceMatrix() {
        double[][] tmp = new double[listNorm.get(0).length][listNorm.size()];
        System.out.println("tmp[][] -> " + tmp.length + " - " + tmp[0].length);
        for (int i = 0; i < listNorm.get(0).length; i++) {
            for (int j = 0; j < listNorm.size(); j++) {
                tmp[i][j] = listNorm.get(j)[i];
            }
        }

        matrixB = new Matrix(tmp); //nFeature x M(jml data)
        //klo data udh 1500 ganti jd
        matrixCovariance = matrixB.transpose().times(matrixB); // [M x nFeature] * [nFeature x M] = M x M
//        matrixCovariance = matrixB.times(matrixB.transpose()); // [M x nFeature] * [nFeature x M] = M x M
    }

    public static void calcEigenValueAndVector() {
        EigenvalueDecomposition eigValDecom = new EigenvalueDecomposition(matrixCovariance);
        double[] eigenValue = eigValDecom.getRealEigenvalues();
        double[][] eigenVec = eigValDecom.getV().getArray();

        HashMap<Integer, Double> indexedEigVal = new HashMap<>();
        for (int i = 0; i < listData.size(); i++) {
            indexedEigVal.put(i, eigenValue[i]);
        }

        List<Map.Entry<Integer, Double>> sortedEigVal = new ArrayList<>(indexedEigVal.entrySet());

        Collections.sort(sortedEigVal, new Comparator<Map.Entry<Integer, Double>>() {
            @Override
            public int compare(Map.Entry<Integer, Double> e1, Map.Entry<Integer, Double> e2) {
                return e2.getValue().compareTo(e1.getValue());
            }
        });

        double[][] tempVector = new double[eigenVec.length][eigenVec[0].length];

        // urut vektor eigen berdasarkan eigen value terbesar. 
        for (int col = 0; col < listData.size(); col++) {
            for (int rows = 0; rows < listData.size(); rows++) {
                tempVector[rows][col] = eigenVec[rows][sortedEigVal.get(col).getKey()]; // key nunjukin kolom
            }
        }

        for (int i = 0; i < eigenVec.length; i++) {
            System.arraycopy(tempVector[i], 0, eigenVec[i], 0, eigenVec[i].length);
        }

        eigenVector = new Matrix(eigenVec); // M x M
    }

    public static void calcEigenface() throws FileNotFoundException, UnsupportedEncodingException {
        //klo data udh 1500 ganti jd
        eigenFace = matrixB.times(eigenVector); // [nFeature x M] * [M x M] = nFeature x M
//        eigenFace = eigenVector.times(matrixB); // [nFeature x M] * [M x M] = nFeature x M

//        PrintWriter writer = new PrintWriter("G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\eigenface_" + gender + ".txt", "UTF-8");
//
//        double[][] arrEigenFace = eigenFace.getArray();
//        for (int i = 0; i < arrEigenFace.length; i++) {
//            for (int j = 0; j < arrEigenFace[0].length; j++) {
//                writer.print(arrEigenFace[i][j] + ";");
//            }
//            writer.println();
//        }
//
//        writer.close();
    }

    public static void calcWeight() throws FileNotFoundException, UnsupportedEncodingException {
        double[] weight;
        String[] tmp;
        double[][] eigenfaceTrans = eigenFace.transpose().getArray(); // M x nFeature
        StringBuilder builder = new StringBuilder();
        listWeight = new ArrayList<>();

        //[K x nFeature] * [nFeature x 1] = [K x 1]
        System.out.println("listNorm size : " + listNorm.size());
        weight = new double[K];
        System.out.println("weight.length : " + weight.length);
        System.out.println("eigenfaceTrans[0].length : " + eigenfaceTrans[0].length);
        for (int i = 0; i < listNorm.size(); i++) {
            weight = new double[K];
            tmp = new String[K+1];
            for (int j = 0; j < weight.length; j++) {
                for (int k = 0; k < eigenfaceTrans[0].length; k++) {
                    weight[j] += eigenfaceTrans[j][k] * listNorm.get(i)[k];
                }
                builder.append(Math.abs(weight[j]) + ",");
                tmp[j] = Math.abs(weight[j])+"";
            }
            tmp[weight.length] = gender+"";
            listWeight.add(tmp);
            builder.append(System.getProperty("line.separator"));
        }
        
        BufferedWriter writer;
        try {
            writer = new BufferedWriter(new FileWriter(HEADER_PRINT+"pca_" + gender + ".csv"));
            writer.write(builder.toString());//save the string representation of the board
            writer.close();
        } catch (IOException ex) {
            System.out.println(ex);
//            Logger.getLogger(Interface.class.getName()).log(Level.SEVERE, null, ex);
        }
//        writer.close();
    }

    public static String[] testing(double[] histImg, String gender) throws FileNotFoundException, UnsupportedEncodingException, IOException {
        nFeature = histImg.length;
        StringBuilder builder = new StringBuilder();
        String path = "";
        String[] weight;

//=============BAGI 255===================
//        for (int j = 0; j < histImg.length; j++) {
//            histImg[j] = histImg[j] / 255;
//        }
//=========NORMALISASI==================
        path = HEADER_PRINT + "\\avg_" + gender + ".csv";
        double[] avgTest = null;
        avgTest = getDataFromCSV(path, histImg.length);
//        BufferedReader in = new BufferedReader(new FileReader("G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\avg_" + gender + ".txt"));
//        while ((line = in.readLine()) != null) {
//            String[] tokens = line.split(";");
//            avgTest = new double[nFeature];
//            for (int i = 0; i < tokens.length; i++) {
//                avgTest[i] = Double.parseDouble(tokens[i]);
//            }
//        }
//        in.close();

        double[] normalizeImg = new double[nFeature];;
        for (int j = 0; j < nFeature; j++) {
            normalizeImg[j] = histImg[j] - avgTest[j];
        }

//========KALI EIGENFACE TIAP GENDER================
        path = HEADER_PRINT + "\\eigenface_" + gender + ".csv";
        Scanner sc = new Scanner(new BufferedReader(new FileReader(path)));
        double[][] eigenface = new double[nFeature][sc.nextLine().trim().split(",").length]; // nFeature x M
        eigenface = getDataFromText2D(path, eigenface.length, eigenface[0].length);
//        BufferedReader in2 = new BufferedReader(new FileReader("G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\eigenface_" + gender + ".txt"));
//        String[] tokens = in2.readLine().split(";");
//        double[][] eigenface = new double[nFeature][tokens.length]; // nFeature x M
//        String line2;
//        int count = 0;
//        while ((line2 = in2.readLine()) != null) {
//            String[] tokens2 = line2.split(";");
//            for (int i = 0; i < tokens2.length; i++) {
//                eigenface[count][i] = Double.parseDouble(tokens2[i]);
//            }
//            count++;
//        }
//        in2.close();

//========CARI WEIGHT================
        double[] weightTest;
        double[][] eigenfaceTransTest = transposeMatrix(eigenface); // M x nFeature

        //[K x nFeature] * [nFeature x 1] = [K x 1]
        weightTest = new double[K];
        weight = new String[K+1];
        for (int j = 0; j < weightTest.length; j++) {
            for (int k = 0; k < eigenfaceTransTest[0].length; k++) {
                weightTest[j] += eigenfaceTransTest[j][k] * normalizeImg[k];
            }
            builder.append(Math.abs(weightTest[j]) + ",");
            weight[j] = Math.abs(weightTest[j])+"";
        }
        weight[weight.length-1] = gender;

        BufferedWriter writer;
        try {
            writer = new BufferedWriter(new FileWriter(HEADER_PRINT + "\\testing.csv"));
            writer.write(builder.toString());//save the string representation of the board
            writer.close();
        } catch (IOException ex) {
            System.out.println(ex);
//            Logger.getLogger(Interface.class.getName()).log(Level.SEVERE, null, ex);
        }
        
//        System.out.println("TESTING DONE!");
        return weight;
    }

    public static double[][] transposeMatrix(double[][] m) {
        double[][] temp = new double[m[0].length][m.length];
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[0].length; j++) {
                temp[j][i] = m[i][j];
            }
        }
        return temp;
    }

    public int getK() {
        return K;
    }

    public void setK(int K) {
        this.K = K;
    }
    
    public ArrayList<String[]> getListWeight() {
        return listWeight;
    }

    //ini gadipake
    public static void calcWeightMatrix() throws FileNotFoundException, UnsupportedEncodingException {
        //[K x nFeature] * [nFeature x 1] = [K x 1]
        Matrix weight = eigenFace.transpose().times(matrixB);
        writeToCSV(weight, "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\pca_" + gender + ".csv");
    }

    //ini gadipake
    public static void calculateWeight() throws FileNotFoundException, UnsupportedEncodingException {
//        PrintWriter writer = new PrintWriter("G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\pca_" + gender + ".txt", "UTF-8");
        double[] weight;
        double[][] eigenfaceTrans = eigenFace.transpose().getArray(); // M x nFeature
        StringBuilder builder = new StringBuilder();

        //[K x nFeature] * [nFeature x 1] = [K x 1]
//        System.out.println("listNorm size " + listNorm.size());
//        System.out.println("listNorm length " + listNorm.get(0).length);
//        System.out.println("k "+K);
//        System.out.println("eigenfaceTrans size " + eigenfaceTrans.length);
//        System.out.println("eigenfaceTrans length " + eigenfaceTrans[0].length);
//        System.out.println("eigfacTrans " + eigenfaceTrans[0][50]);
//        System.out.println("listnorm " + listNorm.get(49)[50]);
        for (int i = 0; i < listNorm.size(); i++) {
            weight = new double[K];
            for (int j = 0; j < weight.length; j++) {
                for (int k = 0; k < eigenfaceTrans[0].length; k++) {
//                    System.out.println("------------------------------------");
//                    System.out.println("j = "+j+" - i = "+i+" - k = "+k);
//                    System.out.println("weight : "+weight[j]);
//                    System.out.println("eigenfaceTrans : "+eigenfaceTrans[j][k]);
//                    System.out.println("listNorm : "+listNorm.get(i)[k]);
                    weight[j] += eigenfaceTrans[i][k] * listNorm.get(i)[k];
//                    System.out.println("Math.abs(weightTest[j]) " + Math.abs(weight[j]));
                }
                builder.append(Math.abs(weight[j]) + ",");
//                writer.print(Math.abs(weight[j]) + ";");
//                System.out.println("");
            }
            builder.append(System.getProperty("line.separator"));
//            writer.println();
//            System.out.println("Finish weight data : " + i + " gender " + gender);
        }

        BufferedWriter writer;
        try {
            writer = new BufferedWriter(new FileWriter("G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\pca_" + gender + ".csv"));
            writer.write(builder.toString());//save the string representation of the board
            writer.close();
        } catch (IOException ex) {
            System.out.println(ex);
//            Logger.getLogger(Interface.class.getName()).log(Level.SEVERE, null, ex);
        }
//        writer.close();
    }

    public static void cobaTest() throws UnsupportedEncodingException, IOException {
        gender = "male";
        String path = "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Dataset\\crop_lfw_male_test\\0_face.jpg";

        SPM_Centrist c = new SPM_Centrist(2);
        c.extract(path);
        System.out.println(c.getHistogram().length) ;
        testing(c.getHistogram(), gender);
    }

    public static void cobaTrain() throws FileNotFoundException, UnsupportedEncodingException {
        gender = "female";
        String path = "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Dataset\\crop_lfw_" + gender;
        File folderTest = new File(path);
        System.out.println(folderTest.getAbsolutePath());
        ArrayList<double[]> listImg = new ArrayList<>();
        int n = 0;
        for (final File fileEntry : folderTest.listFiles()) {
            SPM_Centrist c = new SPM_Centrist(2);
            c.extract(fileEntry.getAbsolutePath());
            listImg.add(c.getHistogram());
            n++;
        }
        listData = listImg;
        nFeature = listData.get(0).length;
        avg = new double[nFeature];

        //AVERAGE
        calcAvg();
        System.out.println("listData size: " + listData.size());
        System.out.println("listData[0] length: " + listData.get(0).length);
        System.out.println("avg length: " + avg.length);
//        for (int i = 0; i < avg.length; i++) {
//            System.out.println(i+". "+avg[i]);
//        }

        //NORMALIZE
        normalize();
        System.out.println("listNorm size : " + listNorm.size());
        System.out.println("listNorm[0] length : " + listNorm.get(0).length);
//        for (int i = 0; i < listNorm.size(); i++) {
//            for (int j = 0; j < listNorm.get(i).length; j++) {
//                System.out.println("i = " + i + " -  j.length = " + listNorm.get(i).length);
//            }
//        }

        //COVARIANCE MATRIX
        calcCovarianceMatrix();
        System.out.println("Matrix B - Row = " + matrixB.getRowDimension());
        System.out.println("Matrix B - Column = " + matrixB.getColumnDimension());
        System.out.println("Matrix Covariance - Row = " + matrixCovariance.getRowDimension());
        System.out.println("Matrix Covariance - Column = " + matrixCovariance.getColumnDimension());

        //EIGEN VALUE & EIGEN VECTOR
        calcEigenValueAndVector();
        System.out.println("eigenVector - row = " + eigenVector.getRowDimension());
        System.out.println("eigenVector - column = " + eigenVector.getColumnDimension());
//        System.out.println("eigen vector : \n" + eigenVector.toString());

        //EIGEN FACE
        calcEigenface();
        System.out.println("eigenFace - row = " + eigenFace.getRowDimension());
        System.out.println("eigenFace - column = " + eigenFace.getColumnDimension());
//        System.out.println("eigenFace : \n" + eigenFace.toString());
        //WEIGHT
        calcWeight();
        System.out.println("Done!");
    }

}
