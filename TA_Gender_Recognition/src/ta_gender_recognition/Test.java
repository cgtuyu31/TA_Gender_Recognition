/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ta_gender_recognition;

import Jama.Matrix;
import Methods.CsvUtils;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import Methods.CsvUtils;
import Methods.FaceDetector;
import Methods.Normalization;
import Methods.PCA;
import Methods.SPM_Centrist;
import Methods.SVMweka;
import Methods.SupportVectorMachine;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import net.semanticmetadata.lire.imageanalysis.features.global.centrist.SpatialPyramidCentrist;
import static ta_gender_recognition.GUI.PATH_HEADER_TRAINING;
import weka.attributeSelection.PrincipalComponents;
import weka.classifiers.functions.SMO;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author Tuyu
 */
public class Test {

    private static String[] classGender = {"male", "female"};
//    private static String[] classGender = {"male"};
    private static String[] pathGenderTrain = {
        GUI.PATH_HEADER_DATASET + "lfw_male",
        GUI.PATH_HEADER_DATASET + "lfw_female"};
    private static String[] pathCropGenderTrain = {
        GUI.PATH_HEADER_DATASET + "crop_lfw_male",
        GUI.PATH_HEADER_DATASET + "crop_lfw_female"};
    private static String[] pathDataCENTRIST = {
        GUI.PATH_HEADER_TRAINING + "train_male.csv",
        GUI.PATH_HEADER_TRAINING + "train_female.csv"};
    private static String[] pathDataCENTRIST2 = {
        GUI.PATH_HEADER_TRAINING + "test_male.csv",
        GUI.PATH_HEADER_TRAINING + "test_female.csv"};
    private static String HEADER_PATH_DATA = GUI.PATH_HEADER_TRAINING;
    private static String pathDataTrain = GUI.PATH_HEADER_TRAINING + "pca_train.csv";
    private static String pathDataTest = GUI.PATH_HEADER_TRAINING + "pca_test_lib.arff";
    private static ArrayList<String[]> modelPCA;
    private static int nData = 1500;
    private static int nTrain = 1300;
    private static int nTest = 200;
    private static int nMale = 0;
    private static int nFemale = 0;
    private static int nTotal = 0;
    private static final int features = 1240;
    private static final int block = 31;
    private static final int pcaK = 40;
    private static final int pcaFeatures = 256;
    private static ArrayList<double[]> dataTrain;
    private static ArrayList<double[]> dataTestMale;
    private static ArrayList<double[]> dataTestFemale;
    private static ArrayList<String[]> dataTest;
    private static double sigma = 0.01;

    public static void cropFace() {
        for (int i = 0; i < classGender.length; i++) {
            File folderGenderTraining = new File(pathGenderTrain[i]);
            for (int j = 0; j < folderGenderTraining.listFiles().length; j++) {
                FaceDetector faceDetector = new FaceDetector();
                Mat[] mats = faceDetector.snipFace(folderGenderTraining.listFiles()[j].toString(), new Size(250, 250));
                for (Mat mat : mats) {
                    Imgcodecs.imwrite(pathCropGenderTrain[i] + "\\" + j + "_face.jpg", mat);
                }
            }
        }
        System.out.println("Detecting Face Done!!");
    }

    public static void getCentristLibData(int gen) throws IOException {
        int n = 0;
        SpatialPyramidCentrist c;

        dataTrain = new ArrayList<>();
        dataTestMale = new ArrayList<>();
        dataTestFemale = new ArrayList<>();

        File folderGenderTraining = new File(pathCropGenderTrain[gen]);
        System.out.println("classGender = " + classGender[gen]);
        nTest = 0;
        for (int i = 0; i < nData; i++) {
            System.out.print(n + ". ");
            BufferedImage image = null;
            image = ImageIO.read(new File(folderGenderTraining.listFiles()[i].toString()));
            c = new SpatialPyramidCentrist();
            c.extract(image);
            if (n < 1300) {
                dataTrain.add(c.getFeatureVector());
            } else if (classGender[gen].equals("male")) {
                dataTestMale.add(c.getFeatureVector());
                nTest++;
            } else {
                dataTestFemale.add(c.getFeatureVector());
                nTest++;
            }
            n++;
        }

        if (gen == 0) {
            nMale = n;
            CsvUtils.writeToCSV(dataTrain, "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\train_male.csv");
            CsvUtils.writeToCSV(dataTestMale, "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\test_male.csv");
        } else {
            nFemale = n;
            CsvUtils.writeToCSV(dataTrain, "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\train_female.csv");
            CsvUtils.writeToCSV(dataTestFemale, "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\test_female.csv");
        }

        nTotal = nMale + nFemale;

        System.out.println("jml dataTrain = " + dataTrain.size());
        System.out.println("jml dataTestMale = " + dataTestMale.size());
        System.out.println("jml dataTestFemale = " + dataTestFemale.size());
        System.out.println("nMale = " + nMale);
        System.out.println("nFemale = " + nFemale);
        System.out.println("nTotal = " + nTotal);
    }

    public static void getTrainTestData(int gen) {
        int n = 0;
        SPM_Centrist c;

        dataTrain = new ArrayList<>();
        dataTestMale = new ArrayList<>();
        dataTestFemale = new ArrayList<>();

        File folderGenderTraining = new File(pathCropGenderTrain[gen]);
        System.out.println("classGender = " + classGender[gen]);
        nTest = 0;
        for (int i = 0; i < nData; i++) {
            System.out.print(n + ". ");
            c = new SPM_Centrist(2);
            c.extract(folderGenderTraining.listFiles()[i].toString());
            if (n < 1300) {
                dataTrain.add(c.getHistogram());
            } else if (classGender[gen].equals("male")) {
                dataTestMale.add(c.getHistogram());
                nTest++;
            } else {
                dataTestFemale.add(c.getHistogram());
                nTest++;
            }
            n++;
        }

        if (gen == 0) {
            nMale = n;
            CsvUtils.writeToCSV(dataTrain, "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\train_male.csv");
            CsvUtils.writeToCSV(dataTestMale, "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\test_male.csv");
        } else {
            nFemale = n;
            CsvUtils.writeToCSV(dataTrain, "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\train_female.csv");
            CsvUtils.writeToCSV(dataTestFemale, "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\test_female.csv");
        }

        nTotal = nMale + nFemale;

        System.out.println("jml dataTrain = " + dataTrain.size());
        System.out.println("jml dataTestMale = " + dataTestMale.size());
        System.out.println("jml dataTestFemale = " + dataTestFemale.size());
        System.out.println("nMale = " + nMale);
        System.out.println("nFemale = " + nFemale);
        System.out.println("nTotal = " + nTotal);
    }

    public static void getTrainTestDataFromCSV(int gen) throws FileNotFoundException {
        dataTrain = CsvUtils.getArrayListDataFromText2D(pathDataCENTRIST[gen], nTrain, features);
//        dataTrain.addAll(getArrayListDataFromText2D(pathDataCENTRIST2[gen], nTest, features));
    }

    public static void getTestData(int gen) {
        int n = 0;
        SPM_Centrist c;

        dataTestMale = new ArrayList<>();
        dataTestFemale = new ArrayList<>();

        File folderGenderTraining = new File(pathCropGenderTrain[gen]);
        System.out.println("classGender = " + classGender[gen]);
        nTest = 0;
        for (int i = nTrain; i < nData; i++) {
            System.out.print(n + ". ");
            c = new SPM_Centrist(2);
            c.extract(folderGenderTraining.listFiles()[i].toString());
            if (classGender[gen].equals("male")) {
                dataTestMale.add(c.getHistogram());
                nTest++;
            } else {
                dataTestFemale.add(c.getHistogram());
                nTest++;
            }
            n++;
        }

        if (gen == 0) {
            nMale = n;
            CsvUtils.writeToCSV(dataTestMale, "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\test_male.csv");
        } else {
            nFemale = n;
            CsvUtils.writeToCSV(dataTestFemale, "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\test_female.csv");
        }

        nTotal = nMale + nFemale;

//        System.out.println("jml dataTrain = " + dataTrain.size());
//        System.out.println("jml dataTestMale = " + dataTestMale.size());
//        System.out.println("jml dataTestFemale = " + dataTestFemale.size());
//        System.out.println("nMale = " + nMale);
//        System.out.println("nFemale = " + nFemale);
//        System.out.println("nTotal = " + nTotal);
    }

    public static void trainPCA() throws FileNotFoundException, UnsupportedEncodingException, IOException {
        modelPCA = new ArrayList<>();
        ArrayList<String[]> listData;
        for (int i = 0; i < classGender.length; i++) {
//        for (int i = 0; i < 1; i++) {
            getCentristLibData(i);
//            getTrainTestData(i);
//            getTrainTestDataFromCSV(i);
            listData = new ArrayList<>();
            for (int j = 0; j < block; j++) {
                System.out.println("======================================================");
                System.out.println("Block : " + (j + 1));
                ArrayList<double[]> tmp = new ArrayList<>();
                for (double[] tes : dataTrain) {
                    tmp.add(Normalization.getChunkArray(tes, pcaFeatures, j));
                }

                PCA pca = new PCA(pcaK, pcaFeatures);
                pca.train(tmp, classGender[i], j);

                ArrayList<String[]> weights = pca.getListWeight();
                CsvUtils.writeStringToCSV(weights, PATH_HEADER_TRAINING + "blocks\\block_" + j + "_" + classGender[i] + ".csv");
                System.out.println("weight.size : " + weights.size());
                System.out.println("weight[0].length : " + weights.get(0).length);
                if (j == 0) {
                    listData.addAll(weights);
                } else {
                    for (int k = 0; k < weights.size(); k++) {
                        String[] tmp1 = listData.get(k);
                        String[] result = new String[(block * pcaK) + 1];
                        System.arraycopy(tmp1, 0, result, 0, tmp1.length);
                        System.arraycopy(weights.get(k), 0, result, j * pcaK, weights.get(k).length);
                        listData.set(k, result);
                    }
                }
                System.out.println("listData.size : " + listData.size());
                System.out.println("listData[0].length : " + listData.get(0).length);
                System.out.println("listData[length-1] : " + listData.get(0)[listData.get(0).length - 1]);
            }
            for (int k = 0; k < listData.size(); k++) {
                listData.get(k)[listData.get(k).length - 1] = classGender[i];
            }
            modelPCA.addAll(listData);
            System.out.println("modelPCA size : " + modelPCA.size());
            System.out.println("modelPCA[].length : " + modelPCA.get(0).length);
        }
        CsvUtils.writeToCSVwithLabel(modelPCA, pathDataTrain);
    }

    public static void trainPCALib() throws FileNotFoundException, Exception {
//        for (int i = 0; i < classGender.length; i++) {
//            getTrainTestDataFromCSV(i);
//            ConverterUtils.DataSource source = new ConverterUtils.DataSource(pathDataCENTRIST[i]);
//            Instances data_train = source.getDataSet();
//
//            PrincipalComponents pca = new PrincipalComponents();
//            pca.setCenterData(false);
//            pca.setVarianceCovered(0.95);
////        pca.setDebug(false);
//            pca.setDoNotCheckCapabilities(false);
//            pca.setInputFormat(data_train);
//
//            Instances newData;
//            newData = pca.useFilter(data_train, pca);
//            System.out.println(newData.get(0).toString().split(",").length);
//            System.out.println("Data :" + newData.size());
//
//            PrintWriter pw = new PrintWriter(new File("hasildataPCA.csv"));
//            StringBuilder sb = new StringBuilder();
//            for (int i = 0; i < newData.size(); i++) {
//                String tampung[] = newData.get(i).toString().split(",");
//                for (int j = 0; j < tampung.length; j++) {
//                    sb.append(tampung[j]);
//                    sb.append(',');
//                }
//                sb.append('\n');
//            }
//            pw.write(sb.toString());
//            pw.close();
//            System.out.println("Save To File Done!");
//        }
    }

    public static void trainSVM() {
//        String[] gender = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"};
//        String[] gender = {"setosa", "versicolor", "virginica"};
        nTotal = nTrain * 2;
//        nTotal = 2643;
//        nTotal = 2700;
//        nTotal = 1300;

        SupportVectorMachine svm = new SupportVectorMachine();
        svm.setSigma(sigma);
        try {
            String[][] datatrain = CsvUtils.getDataStringFromText2D(pathDataTrain, (nTotal + 1), (features + 1));
            double[][] data = new double[nTotal][features];
            for (int i = 1; i < datatrain.length; i++) {
                for (int j = 0; j < datatrain[0].length - 1; j++) {
                    data[i - 1][j] = Double.parseDouble(datatrain[i][j]);
//                    System.out.print(data[i - 1][j] + " ");
                }
//                System.out.println("");
            }

            for (int index = 0; index < classGender.length; index++) {
                double[] classList = new double[nTotal + 1];
                for (int i = 0; i < classList.length - 1; i++) {
                    if (datatrain[i + 1][features].equals(classGender[index])) {
                        classList[i] = 1;
                    } else {
                        classList[i] = -1;
                    }
                }
                classList[nTotal] = 0;

                //create RBF Matrix
                double[][] rbfMatrix = svm.createRBFMatrix(data);
                double[][] linearEquation = svm.createLinearEquationMatrix(rbfMatrix, classList);

                Matrix solutions = svm.getSolutions(linearEquation, classList);
                //print solutions
                for (int i = 0; i < linearEquation.length; i++) {
                    System.out.println("X - " + (i + 1) + " : " + solutions.get(i, 0));
                }

                System.out.println("Model for " + classGender[index] + " with " + features + " feature is saved!");
                StringBuilder builder = new StringBuilder();
                for (int i = 0; i < linearEquation.length; i++) {
                    builder.append(solutions.get(i, 0));
                    builder.append(System.getProperty("line.separator"));
                }

                BufferedWriter writer;
                try {
                    writer = new BufferedWriter(new FileWriter("G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\svm_" + features + "_sigma-" + svm.getSigma() + "_model-" + classGender[index] + ".txt"));
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

    public static void trainSVMsmo() throws Exception {
        ConverterUtils.DataSource src = new ConverterUtils.DataSource(pathDataTrain);
        Instances data_train = src.getDataSet();
        data_train.setClass(data_train.attribute("class"));

        SMO smo = new SMO();

        //train and build classifier
        RBFKernel rbf = new RBFKernel();
        rbf.setGamma(sigma);
        smo.setKernel(rbf);
        smo.buildClassifier(data_train);

        src = new ConverterUtils.DataSource(pathDataTest);
        Instances data_test = src.getDataSet();
        data_test.setClass(data_train.attribute("class"));
        //test
        Evaluation eva = new Evaluation(data_train);
        eva.evaluateModel(smo, data_test);
        System.out.println("Correctly Classified Instances :  " + eva.pctCorrect());
        System.out.println("Correctly Classified Instances :  " + eva.pctIncorrect());
        double[][] confMatrix = eva.confusionMatrix();
        System.out.println("male \t female");
        for (int i = 0; i < confMatrix.length; i++) {
            for (int j = 0; j < confMatrix[0].length; j++) {
                System.out.print(confMatrix[i][j] + " \t ");
            }
            System.out.println("");
        }

//        SVMweka svm = new SVMweka();
//
//        svm.loadCSV(pathDataTrain);
//        svm.buildModel();
//        svm.saveModelToFile("G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\svm_model.model");
//        
//        
//        svm.loadModelFromFile("G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\svm_model.model");
//        svm.predictForRow("1,5,10",",");
//        
//        Instances ins = svm.getInstances();
//        int n = 1;
//        for (Instance in : ins) {
//            System.out.println("Predict Value Data - " + n + " : " + svm.classifyInstance(in));
//            n++;
//        }
    }

    public static void testPCA() throws UnsupportedEncodingException, IOException {
//        getTestData(0);
//        getTestData(1);
        dataTestMale = CsvUtils.getArrayListDataFromText2D(HEADER_PATH_DATA + "test_male.csv", nTest, 7936);
        dataTestFemale = CsvUtils.getArrayListDataFromText2D(HEADER_PATH_DATA + "test_female.csv", nTest, 7936);
        dataTest = new ArrayList<>();
        PCA pca;
        String[] weightTest;
        for (int i = 0; i < dataTestMale.size(); i++) {
//            dataTest.add(pca.testing(dataTestMale.get(i), classGender[0]));
            weightTest = new String[(block * pcaK) + 1];
            for (int j = 0; j < block; j++) {
                double[] tmp = new double[pcaFeatures];
                tmp = Normalization.getChunkArray(dataTestMale.get(i), pcaFeatures, j);

                pca = new PCA(pcaK, pcaFeatures);
                String[] weight = pca.testing(tmp, classGender[0], j);
                System.arraycopy(weight, 0, weightTest, j * pcaK, weight.length);
                weightTest[weightTest.length - 1] = classGender[0];
            }
            dataTest.add(weightTest);
        }
        for (int i = 0; i < dataTestFemale.size(); i++) {
//            dataTest.add(pca.testing(dataTestFemale.get(j), classGender[1]));
            weightTest = new String[(block * pcaK) + 1];
            for (int j = 0; j < block; j++) {
                double[] tmp = new double[pcaFeatures];
                tmp = Normalization.getChunkArray(dataTestFemale.get(i), pcaFeatures, j);

                pca = new PCA(pcaK, pcaFeatures);
                String[] weight = pca.testing(tmp, classGender[1], j);
                System.arraycopy(weight, 0, weightTest, j * pcaK, weight.length);
                weightTest[weightTest.length - 1] = classGender[1];
            }
            dataTest.add(weightTest);
        }

        CsvUtils.writeToCSVwithLabel(dataTest, pathDataTest);
    }

    public static void testSVM() {
        String[] gender = {"male", "female"};
        SupportVectorMachine svm = new SupportVectorMachine();
        try {
            String[][] dataset = CsvUtils.getDataStringFromText2D(pathDataTrain, (nTotal + 1), (features + 1));
            double[][] data = new double[nTotal][features];

            for (int i = 1; i < dataset.length; i++) {
                for (int j = 0; j < dataset[0].length - 1; j++) {
                    data[i - 1][j] = Double.parseDouble(dataset[i][j]);
                }
            }

            double truePositiveRateAvg = 0;

            for (int index = 0; index < gender.length; index++) {

                double[] classList = new double[(nTotal + 1)];
                for (int i = 0; i < classList.length - 1; i++) {
                    if (dataset[i][features].equals(gender[index])) {
                        classList[i] = 1;
                    } else {
                        classList[i] = -1;
                    }
                }
                classList[nTotal] = 0;

                //get model (alpha and bias)
                String modelPath = "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\svm_" + features + "_sigma-" + sigma + "_model-" + gender[index] + ".txt";

                double[] solutions = CsvUtils.getDataFromText(modelPath, (nTotal + 1));

                //get testing data, create RBF matrix test
                String[][] datasetTest = CsvUtils.getDataStringFromText2D(pathDataTest, (nTest * 2 + 1), (features + 1));
                double[][] dataTest = new double[nTest * 2][features];

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
                    end = nTest;
                } else if (gender[index].equals("female")) {
                    start = nTest;
                    end = nTest * 2;
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
                System.out.println("------------------------------------------------------------------------");
                truePositiveRateAvg += (trueClassified / (nTest / classGender.length));
//              System.out.println("Total False Classified : " + falseClassified);

            }
            System.out.println("True Positive Rate Avg: " + (truePositiveRateAvg / classGender.length));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void SVM_Test_Confusion(Instances data_train, Instances data_test, double sigma, int countFeature, String model_path) throws FileNotFoundException {
//        String[] classes = {"male", "female"};
//        //String[] classes = {"normal", "lain"};
//        double[][] tempData_train = new double[data_train.size()][countFeature];
//        double[][] tempData_test = new double[data_test.size()][countFeature];
//        String[] tempLabel_train = new String[data_train.size()];
//        String[] tempLabel_test = new String[data_test.size()];
//        double[] classList = new double[data_train.size() + 1];
//        classList[data_train.size()] = 0.0;
//        double[][] svmResult = new double[data_test.size()][classes.length];
//        SupportVectorMachine svm = new SupportVectorMachine();
//
//        //data train
//        for (int i = 0; i < data_train.size(); i++) {
//            String tempL_train[] = data_train.get(i).toString().split(",");
//            tempLabel_train[i] = tempL_train[countFeature];
//            for (int j = 0; j < countFeature; j++) {
//                String tempS[] = data_train.get(i).toString().split(",");
//                tempData_train[i][j] = Double.parseDouble(tempS[j]);
//            }
//
//        }
//
//        double truePositiveRateAvg[] = new double[5];
//        double trueClassified[] = new double[5];
//        double falseClassified[] = new double[5];
//        int countLabel[] = new int[5];
//        for (int i = 0; i < 5; i++) {
//            countLabel[i] = 0;
//            truePositiveRateAvg[i] = 0;
//            trueClassified[i] = 0;
//            falseClassified[i] = 0;
//        }
//
//        //data train ambil labelnya sesuai index ( 5 kelas )
//        for (int index = 0; index < classes.length; index++) {
//            for (int i = 0; i < classList.length - 1; i++) {
//                if (tempLabel_train[i].equals(classes[index])) {
//                    //System.out.println("index : "+tempLabel_train[i]);
//                    classList[i] = 1.0;
//                } else {
//                    //System.out.println("bukan index : "+tempLabel_train[i]);
//                    classList[i] = -1.0;
//                }
//            }
//
//            String modelPath = model_path + sigma + "_model-" + classes[index] + ".txt";
//            double[] solutions = getDataFromText(modelPath, data_train.size() + 1);
//
//            //get testing data, create RBF matrix test
//            for (int i = 0; i < data_test.size(); i++) {
//                String tempL[] = data_test.get(i).toString().split(",");
//                tempLabel_test[i] = tempL[countFeature];
//                for (int j = 0; j < countFeature; j++) {
//                    String tempS[] = data_test.get(i).toString().split(",");
//                    tempData_test[i][j] = Double.parseDouble(tempS[j]);
//                }
//            }
//
//            for (int i = 0; i < data_test.size(); i++) {
//                if (tempLabel_test[i].equals(classes[index])) {
//                    double[] rbfMatrixTest = svm.createRBFTestMatrix(tempData_train, sigma, tempData_test[i]);
//                    double result = svm.classify(solutions, rbfMatrixTest, classList);
//                    //System.out.println("result : " + result);
//                    int temp = 0;
//                    switch (tempLabel_test[i]) {
//                        case "male":
//                            temp = 1;
//                            break;
//                        case "female":
//                            temp = 0;
//                            break;
//                    }
//                    svmResult[i][temp] = result;
//                    //System.out.println("row " + i + " isi " + temp + " :" + result);
//                }
//            }
//        }
//
//        int[][] confusionMatrix = svm.createConfusionMatrix(svmResult, 2, tempLabel_test);
//        System.out.println("Confusion Matrix");
//        for (int i = 0; i < confusionMatrix.length; i++) {
//            for (int j = 0; j < confusionMatrix.length; j++) {
//                System.out.print(confusionMatrix[i][j] + " ");
//            }
//            System.out.println("" + classes[i]);
//        }
//        System.out.println("\nAccuracy : " + svm.calculateAccuracy(confusionMatrix) * 100 + "%");

        //BUAT CEK ISI SVMRESULT
//        for (int i = 0; i < data_test.size(); i++) {
//            System.out.print("data ke " + i);
//            for (int j = 0; j < 5; j++) {
//                System.out.print("||" + svmResult[i][j] + ",");
//            }
//            System.out.println("");
//        }
    }

    public static void main(String arg[]) throws FileNotFoundException, UnsupportedEncodingException, IOException, Exception {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        cropFace();
//        trainPCA();
//        testPCA();
        trainSVMsmo();
//        testSVM();

//        ConverterUtils.DataSource source = new ConverterUtils.DataSource(pathDataTrain);
//        Instances data_train = source.getDataSet();
//        source = new ConverterUtils.DataSource(pathDataTest);
//        Instances data_test = source.getDataSet();
//        SVM_Test_Confusion(data_train, data_test, sigma, features, 
//                "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\svm_"+features+"_sigma-");
//        trainSVMsmo();
//        SPM_Centrist cc = new SPM_Centrist(2);
//        try {
//            for (int i = 0; i < classGender.length; i++) {
//                File folderGenderTraining = new File(pathGenderTrain[i]);
//                for (final File fileEntry : folderGenderTraining.listFiles()) {
//                    FaceDetector faceDetector = new FaceDetector();
//                    System.out.println(fileEntry.getAbsolutePath());
//                    Mat[] mats = faceDetector.snipFace(fileEntry.getAbsolutePath(), new Size(250, 250));
//                    for (Mat mat : mats) {
//                        Imgcodecs.imwrite(pathCropGenderTrain + "\\" + i + "_face.jpg", mat);
//                    }
//                }
//                listImg.clear();
//                folderGenderTraining = new File(pathCropGenderTrain[i]);
//                for (final File fileEntry : folderGenderTraining.listFiles()) {
//                    System.out.println(fileEntry.getAbsolutePath());
//                    SPM_Centrist c = new SPM_Centrist(2);
//                    c.extract(fileEntry.getAbsolutePath());
//                    listImg.add(c.getHistogram());
//                    PCA pca = new PCA();
//                    pca.train(listImg, classGender[i]);
//                }
//            }
//        } catch (Exception e) {
//            System.out.println("File not found");
//            System.out.println(e);
//        }
//        SVM_Lama svm = new SVM_Lama();
//        try {
//            svm.trainGender(classGender[0], classGender[1]);
//            System.out.println("Model SVM [Ekspresi " + classGender[0] + "] Succesfully Created\n");
//            svm.trainGender(classGender[1], classGender[0]);
//            System.out.println("Model SVM [Ekspresi " + classGender[1] + "] Succesfully Created\n");
//
//        } catch (Exception e) {
//            e.printStackTrace();
//            System.out.println("Error when create SVM Model");
//        }
//        cobaReadCsv();'
//    double[][] test = new double [200000][42];
//        System.out.println(test.length);
//        System.out.println(test[0].length);
    }

    public static void cobaReadCsv() throws FileNotFoundException {
        String path = "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\pca_read.csv";
//        path = "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\avg_male.csv";
        double[][] data = CsvUtils.getDataFromText2D(path, 7, 4);
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                System.out.print(data[i][j] + " ");
            }
            System.out.println("");
        }
    }

}
