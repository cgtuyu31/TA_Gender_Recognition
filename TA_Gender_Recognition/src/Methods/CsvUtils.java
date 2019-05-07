/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Methods;

import Jama.Matrix;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

/**
 *
 * @author Tuyu
 */
public class CsvUtils {

    public static void writeToCSVwithLabel(ArrayList<String[]> lists, String path) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < lists.get(0).length; i++) {
            if (i != lists.get(0).length - 1) {
                builder.append("nilai " + (i + 1) + ",");
            } else {
                builder.append("class");
            }
        }
        builder.append(System.getProperty("line.separator"));
        for (int i = 0; i < lists.size(); i++) {
            for (int j = 0; j < lists.get(i).length; j++) {
                if (j == lists.get(0).length - 1) {
                    builder.append(lists.get(i)[j]);
                } else {
                    builder.append(lists.get(i)[j] + ",");
                }
            }
            builder.append(System.getProperty("line.separator"));
        }

        BufferedWriter writer;
        try {
            writer = new BufferedWriter(new FileWriter(path));
            writer.write(builder.toString());//save the string representation of the board
            writer.close();
        } catch (IOException ex) {
            System.out.println(ex);
//            Logger.getLogger(Interface.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public static void writeAListStringToCSV(ArrayList<String[]> lists, String path) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < lists.size(); i++) {
            for (int j = 0; j < lists.get(0).length; j++) {
                builder.append(lists.get(i)[j] + ",");
            }
            builder.append(System.getProperty("line.separator"));
        }

        BufferedWriter writer;
        try {
            writer = new BufferedWriter(new FileWriter(path));
            writer.write(builder.toString());//save the string representation of the board
            writer.close();
        } catch (IOException ex) {
            System.out.println(ex);
//            Logger.getLogger(Interface.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public static void writeAListDoubleToCSV(ArrayList<double[]> lists, String path) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < lists.size(); i++) {
            for (int j = 0; j < lists.get(0).length; j++) {
                builder.append(lists.get(i)[j] + ",");
            }
            builder.append(System.getProperty("line.separator"));
        }

        BufferedWriter writer;
        try {
            writer = new BufferedWriter(new FileWriter(path));
            writer.write(builder.toString());//save the string representation of the board
            writer.close();
        } catch (IOException ex) {
            System.out.println(ex);
//            Logger.getLogger(Interface.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void writeDoubleToCSV(double[] lists, String path) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < lists.length; i++) {
            builder.append(lists[i] + ",");
        }

        BufferedWriter writer;
        try {
            writer = new BufferedWriter(new FileWriter(path));
            writer.write(builder.toString());//save the string representation of the board
            writer.close();
        } catch (IOException ex) {
            System.out.println(ex);
//            Logger.getLogger(Interface.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void writeStringToCSV(String[] lists, String path) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < lists.length; i++) {
            builder.append(lists[i]);
            builder.append(System.getProperty("line.separator"));
        }

        BufferedWriter writer;
        try {
            writer = new BufferedWriter(new FileWriter(path));
            writer.write(builder.toString());//save the string representation of the board
            writer.close();
        } catch (IOException ex) {
            System.out.println(ex);
//            Logger.getLogger(Interface.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public static void writeMatrixToCSV(Matrix lists, String path) {
        double[][] tmp = lists.getArray();
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < tmp.length; i++) {
            for (int j = 0; j < tmp[0].length; j++) {
                builder.append(tmp[i][j] + ",");
            }
            builder.append(System.getProperty("line.separator"));
        }

        BufferedWriter writer;
        try {
            writer = new BufferedWriter(new FileWriter(path));
            writer.write(builder.toString());//save the string representation of the board
            writer.close();
        } catch (IOException ex) {
            System.out.println(ex);
//            Logger.getLogger(Interface.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    // n= baris, features = kolom
    public static double[][] getDataFromText(String path, int n, int features) throws FileNotFoundException {
        Scanner sc = new Scanner(new BufferedReader(new FileReader(path)));
        double[][] data = new double[n][features];
        while (sc.hasNextLine()) {
            for (int i = 0; i < data.length; i++) {
                String[] line = sc.nextLine().trim().split(",");
                for (int j = 0; j < line.length; j++) {
                    data[i][j] = Double.parseDouble(line[j]);
                }
            }
        }
        return data;
    }

    public static ArrayList<double[]> getAListDataFromText(String path, int n, int features) throws FileNotFoundException {
        Scanner sc = new Scanner(new BufferedReader(new FileReader(path)));
//        double[][] data = new double[n][features];
        ArrayList<double[]> data = new ArrayList<>();
        while (sc.hasNextLine()) {
            for (int i = 0; i < n; i++) {
                String[] line = sc.nextLine().trim().split(",");
                double[] tmp = new double[line.length];
                for (int j = 0; j < line.length; j++) {
                    tmp[j] = Double.parseDouble(line[j]);
                }
                data.add(tmp);
            }
        }
        return data;
    }

    // n= baris, features = kolom
    public static String[][] getDataStringFromText(String path, int n, int features) throws FileNotFoundException {
        Scanner sc = new Scanner(new BufferedReader(new FileReader(path)));
        String[][] data = new String[n][features];
        while (sc.hasNextLine()) {
            for (int i = 0; i < data.length; i++) {
                String[] line = sc.nextLine().trim().split(",");
                for (int j = 0; j < line.length; j++) {
                    data[i][j] = line[j];
                }
            }
        }
        return data;
    }

    public static double[] getDataFromText(String path, int n) throws FileNotFoundException {
        Scanner sc = new Scanner(new BufferedReader(new FileReader(path)));
        double[] data = new double[n];
        for (int i = 0; i < data.length; i++) {
            String[] line = sc.nextLine().trim().split(",");
            for (int j = 0; j < line.length; j++) {
                data[i] = Double.parseDouble(line[j]);
            }
        }
        return data;
    }
    
    public static double[] getDataFromCSV(String path, int n) throws FileNotFoundException, IOException {
        BufferedReader reader = new BufferedReader(new FileReader(path));
        String line;
        double[] data = new double[n];
        while ((line = reader.readLine()) != null) {
            String[] tokens = line.split(",");
            for (int j = 0; j < tokens.length; j++) {
                data[j] = Double.parseDouble(tokens[j]);
            }
        }
        reader.close();
        return data;
    }
}
