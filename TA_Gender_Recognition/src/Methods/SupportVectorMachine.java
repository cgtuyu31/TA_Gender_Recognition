/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Methods;

import Jama.Matrix;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.opencv.core.Core;
import static Methods.CsvUtils.getDataFromText;
import static Methods.CsvUtils.getDataStringFromText2D;
import static Methods.CsvUtils.writeToCSV;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Debug;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author Tuyu
 */
public class SupportVectorMachine {

    static final int features = 1240;
    static int n = 2600;
    static double sigma = 10;

    //test
    public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException, IOException, Exception {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        cobaTrainLibSVM();
//        cobaTrain();
//        cobaTest();
    }

    public static void cobaTrain() {
        //class -> 1 = male, 0 = female
        String[] gender = {"male", "female"};
        String path = "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\pca_train.csv";
//        String[] gender = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"};
//        String[] gender = {"setosa", "versicolor", "virginica"};

        SupportVectorMachine svm = new SupportVectorMachine();
        svm.setSigma(sigma);
        try {
            String[][] datatrain = getDataStringFromText2D(path, (n + 1), (features + 1));
            double[][] data = new double[n][features];
            for (int i = 1; i < datatrain.length; i++) {
                for (int j = 0; j < datatrain[0].length - 1; j++) {
                    data[i - 1][j] = Double.parseDouble(datatrain[i][j]);
//                    System.out.print(data[i - 1][j] + " ");
                }
//                System.out.println("");
            }

            for (int index = 0; index < gender.length; index++) {
                double[] classList = new double[n + 1];
                for (int i = 0; i < classList.length - 1; i++) {
                    if (datatrain[i + 1][features].equals(gender[index])) {
                        classList[i] = 1;
                    } else {
                        classList[i] = -1;
                    }
                }
                classList[n] = 0;

                //create RBF Matrix
                double[][] rbfMatrix = svm.createRBFMatrix(data);
                double[][] linearEquation = svm.createLinearEquationMatrix(rbfMatrix, classList);

                Matrix solutions = svm.getSolutions(linearEquation, classList);
                //print solutions
                for (int i = 0; i < linearEquation.length; i++) {
                    System.out.println("X - " + (i + 1) + " : " + solutions.get(i, 0));
                }

                System.out.println("Model for " + gender[index] + " with " + features + " feature is saved!");
                StringBuilder builder = new StringBuilder();
                for (int i = 0; i < linearEquation.length; i++) {
                    builder.append(solutions.get(i, 0));
                    builder.append(System.getProperty("line.separator"));
                }

                BufferedWriter writer;
                try {
                    writer = new BufferedWriter(new FileWriter("G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\svm_" + features + "_sigma-" + svm.getSigma() + "_model-" + gender[index] + ".txt"));
                    writer.write(builder.toString());//save the string representation of the board
                    writer.close();
                } catch (IOException e) {
                    System.out.println(e);
//                    Logger.getLogger(TrainInterface.class.getName()).log(Level.SEVERE, null, ex);
                }

            }
            System.out.println("CREATE MODEL SVM DONE!");
//            JOptionPane.showMessageDialog(null, "Model SVM Done", "InfoBox: SVM Done ", JOptionPane.INFORMATION_MESSAGE);

        } catch (FileNotFoundException e) {
            System.out.println(e);
//            Logger.getLogger(TrainInterface.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void cobaTest() {
        String[] gender = {"male", "female"};
        String path = "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\pca_train.csv";
        SupportVectorMachine svm = new SupportVectorMachine();
        try {
            String[][] dataset = getDataStringFromText2D(path, (n + 1), (features + 1));
            double[][] data = new double[n][features];

            for (int i = 1; i < dataset.length; i++) {
                for (int j = 0; j < dataset[0].length - 1; j++) {
                    data[i - 1][j] = Double.parseDouble(dataset[i][j]);
                }
            }

            double truePositiveRateAvg = 0;

            for (int index = 0; index < gender.length; index++) {

                double[] classList = new double[(n + 1)];
                for (int i = 0; i < classList.length - 1; i++) {
                    if (dataset[i][features].equals(gender[index])) {
                        classList[i] = 1;
                    } else {
                        classList[i] = -1;
                    }
                }
                classList[1680] = 0;

                //get model (alpha and bias)
                String modelPath = "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\svm_" + features + "_sigma-" + sigma + "_model-" + gender[index] + ".txt";

                double[] solutions = getDataFromText(modelPath, (n + 1));

                //get testing data, create RBF matrix test
                String pathTest = "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\testing.csv";
                String[][] datasetTest = getDataStringFromText2D(pathTest, n + 1, (features + 1));
                double[][] dataTest = new double[n][features];

                for (int i = 1; i < datasetTest.length; i++) {
                    for (int j = 0; j < datasetTest[0].length - 1; j++) {
                        dataTest[i - 1][j] = Double.parseDouble(datasetTest[i][j]);
                    }
                }

                double trueClassified = 0;
                double falseClassified = 0;

                int start = 0, end = 0;
                if (gender[index].equals("male")) {
                    start = 0;
                    end = 1316;
                } else if (gender[index].equals("female")) {
                    start = 1316;
                    end = 2643;
                } else {
                    System.out.println("ERROR CLASS!!");
                }

                for (int i = start; i < end; i++) {
                    double[] rbfMatrixTest = svm.createRBFTestMatrix(data, sigma, dataTest[i]);

                    //classify
                    double result = svm.classify(solutions, rbfMatrixTest, classList);
                    if (Math.signum(result) == 1) {
                        trueClassified++;
                    } else {
                        falseClassified++;
                    }
                    System.out.println("Result Signum " + Math.signum(result));
                    System.out.println("Result of classification: " + result);
                    System.out.println("====================================");
                }
                System.out.println("Total True Classified : " + trueClassified + " for " + gender[index]);
                truePositiveRateAvg += (trueClassified / 30);
//              System.out.println("Total False Classified : " + falseClassified);

            }
            System.out.println("True Positive Rate Avg: " + (truePositiveRateAvg / 24));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void cobaTrainLibSVM() throws FileNotFoundException, Exception {
//        String path = "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\pca_all.csv";
//        String MODELPATH = "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\model.csv";
//
//        ModelGenerator mg = new ModelGenerator();
//
//        Instances dataset = mg.loadDataset(path);
//
//        Filter filter = new Normalize();
//
//        // divide dataset to train dataset 80% and test dataset 20%
//        int trainSize = (int) Math.round(dataset.numInstances() * 0.8);
//        int testSize = dataset.numInstances() - trainSize;
//
//        dataset.randomize(
//                new Debug.Random(1));// if you comment this line the accuracy of the model will be droped from 96.6% to 80%
//
//        //Normalize dataset
//        filter.setInputFormat(dataset);
//        Instances datasetnor = Filter.useFilter(dataset, filter);
//
//        Instances traindataset = new Instances(datasetnor, 0, trainSize);
//        Instances testdataset = new Instances(datasetnor, trainSize, testSize);
//
//        // build classifier with train dataset             
//        MultilayerPerceptron ann = (MultilayerPerceptron) mg.buildClassifier(traindataset);
//
//        // Evaluate classifier with test dataset
//        String evalsummary = mg.evaluateModel(ann, traindataset, testdataset);
//
//        System.out.println(
//                "Evaluation: " + evalsummary);
//
//        //Save model 
//        mg.saveModel(ann, MODELPATH);
//
//        //classifiy a single instance 
//        ModelClassifier cls = new ModelClassifier();
//        String classname = cls.classifiy(Filter.useFilter(cls.createInstance(1.6, 0.2, 0), filter), MODELPATH);
//
//        System.out.println(
//                "\n The class name for the instance with petallength = 1.6 and petalwidth =0.2 is  " + classname);

    }

    private double calculateRBFKernel(double[][] data, double sigma, int classSource, int classTarget) {
        double value = 0;
        for (int i = 0; i < data[0].length; i++) {
            value += Math.pow(data[classSource][i] - data[classTarget][i], 2);
        }
        return Math.exp(-(value) / (2 * Math.pow(sigma, 2)));
    }

    public double[][] createRBFMatrix(double data[][]) {
        double[][] rbfMatrix = new double[data.length][data.length];
        for (int i = 0; i < rbfMatrix.length; i++) {
            for (int j = 0; j < rbfMatrix.length; j++) {
                rbfMatrix[i][j] = calculateRBFKernel(data, sigma, i, j);
            }
        }
        return rbfMatrix;
    }

    public double[][] createLinearEquationMatrix(double[][] rbfMatrix, double[] classList) {
        double[][] linearEquationMatrix = new double[rbfMatrix.length + 1][rbfMatrix.length + 1];

        for (int i = 0; i < rbfMatrix.length; i++) {
            for (int j = 0; j < rbfMatrix.length; j++) {
                linearEquationMatrix[i][j] = rbfMatrix[i][j] * classList[j];
            }
        }

        for (int i = 0; i < linearEquationMatrix.length; i++) {
            for (int j = 0; j < linearEquationMatrix.length; j++) {
                //untuk inisialisasi koefisien bias
                if (i == linearEquationMatrix.length - 1) {
                    linearEquationMatrix[i][linearEquationMatrix.length - 1] = 0;
                } else {
                    linearEquationMatrix[i][linearEquationMatrix.length - 1] = 1;
                }

                //tambah persamaan untuk class
                linearEquationMatrix[linearEquationMatrix.length - 1][j] = classList[j];
            }
        }
        return linearEquationMatrix;
    }

    public Matrix getSolutions(double[][] linearEquationMatrix, double[] classList) {

        //matriks persamaan linear (kiri)
//        RealMatrix linearEquation = new Array2DRowRealMatrix(linearEquationMatrix);
//        DecompositionSolver solver = new LUDecomposition(linearEquation).getSolver(); 
//        RealVector constants = new ArrayRealVector(classList);
//        RealVector solutions = solver.solve(constants); 
        Matrix linearEquation = new Matrix(linearEquationMatrix);
        //matriks persamaan linear (kanan)
        Matrix contants = new Matrix(classList, classList.length);
        //solusi persamaan linear -> alpha1, alpha2, alpha3, ..., alpha-n, bias
        Matrix solutions = linearEquation.solve(contants);
        return solutions;
    }

    public double[] createRBFTestMatrix(double data[][], double sigma, double[] test) {
        double[] rbfMatrixUji = new double[data.length];
        double temp = 0;
        for (int i = 0; i < data.length; i++) {
            temp = 0;
            for (int j = 0; j < data[0].length; j++) {
                temp += Math.pow(test[j] - data[i][j], 2);
                //System.out.println("(" + test[j] + " - " + data[i][j] + ")^2 = " + temp);
            }
            rbfMatrixUji[i] = Math.exp(-(temp) / (2 * Math.pow(sigma, 2)));
//            System.out.println("---------------------------");
        }
        return rbfMatrixUji;
    }

    //decision function
    public double classify(double[] solutions, double[] rbfTest, double[] classList) {
        //f(x) = sign(sum of alpha-i*Yi*K(X,Xi) + b)
        double value = 0;
        //Jumlah dari perkalian alpha-i dengan K(X,Xi) dengan Yi
        for (int i = 0; i < classList.length - 1; i++) {
            value += solutions[i] * rbfTest[i] * classList[i];
//            System.out.println(solutions[i]+" * "+rbfTest[i]+" * "+classList[i]);
        }

        //jumlah perkalian diatas ditambah dengan bias
        value += solutions[classList.length - 1];

        //System.out.println("Value = " + value);
        //return Math.signum(value);
        return value;
    }

    public void printMatrix(double[][] matriks) {
        for (int i = 0; i < matriks.length; i++) {
            for (int j = 0; j < matriks.length; j++) {
                System.out.print(matriks[i][j] + "\t");
            }
            System.out.println("");
        }
    }

    public static int findMaxValue(double[] data) {
        double max = 0;
        int index = 0;

        for (int i = 0; i < data.length; i++) {
            if (data[i] > max) {
                max = data[i];
                index = i;
            }
        }

        //System.out.println("Data : " + data[index] + " at index -> " + index);
        return index;
    }

    public static double calculateAccuracy(int[][] confusionMatrix) {
        double accuracy = 0;
        double truePositive = 0;
        double totalData = 0;

        for (int i = 0; i < confusionMatrix.length; i++) {
            for (int j = 0; j < confusionMatrix.length; j++) {
                totalData += confusionMatrix[i][j];
                if (i == j) {
                    truePositive += confusionMatrix[i][j];
                }
            }
        }

        accuracy = truePositive / totalData;

        return accuracy;
    }

    public static int[][] createConfusionMatrix(double[][] data) {
        int[][] confusionMatrix = new int[24][24];
        int predictedIndex = 0;

        for (int i = 0; i < data.length; i++) {
            //System.out.println("index: " + (i / 5));
            predictedIndex = findMaxValue(data[i]);
            confusionMatrix[i / (data.length / 24)][predictedIndex]++;
        }

        return confusionMatrix;
    }
    
    public static int[][] createConfusionMatrix(double[][] data, int countLabel, String[] tempLabel_test) {
        int[][] confusionMatrix = new int[countLabel][countLabel];
        int predictedIndex = 0;

        for (int i = 0; i < data.length; i++) {
            //System.out.println("index: " + (i / 5));
            predictedIndex = findMaxValue(data[i]);
            int temp = ubahLabelJadiInt(tempLabel_test[i]);
            confusionMatrix[temp][predictedIndex]++;
        }

        return confusionMatrix;
    }
    
    public static int ubahLabelJadiInt(String label) {
        int result = 0;
        switch (label) {
            case "male":
                result = 1;
                break;
            case "female":
                result = 0;
                break;
        }
        return result;
    }
    
    public double getSigma() {
        return sigma;
    }

    public void setSigma(double sigma) {
        this.sigma = sigma;
    }
}
