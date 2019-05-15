/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Methods;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import ta_gender_recognition.GUI;
import static ta_gender_recognition.GUI.classGender;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author Tuyu
 */
public class Test_DatasetManual {

    private static String[] pathCropGenderTrain = {
        GUI.PATH_HEADER_DATASET + "manual_crop_male",
        GUI.PATH_HEADER_DATASET + "manual_crop_female"};
    private static String[] pathGenderTrain = {
        GUI.PATH_HEADER_DATASET + "manual_male",
        GUI.PATH_HEADER_DATASET + "manual_female"};
    private static int nData = 76;
    private static String PATH_TRAINING = GUI.PATH_HEADER_TRAINING + "centrist\\k-40\\";
    private static String PATH_TRAINING_BHEP = GUI.PATH_HEADER_TRAINING + "centristBHEP\\k-40\\";
    private static int block = 31;
    private static int nTest = 76;
    private static ArrayList<double[]> dataTestMale;
    private static ArrayList<double[]> dataTestFemale;
    private static int totFeatures = 7936;
    private static int pcaK = 40;
    private static final int pcaFeatures = 256;
    private static double sigma = 0.01;

    public static void centrist() {
        int n;
        SPM_Centrist c = new SPM_Centrist(2);
        c.setType(0);
        ArrayList<double[]> dataMale = new ArrayList<>();
        ArrayList<double[]> dataFemale = new ArrayList<>();

        for (int g = 0; g < classGender.length; g++) {
            n = 0;
            File folderGenderTraining = new File(pathCropGenderTrain[g]);
            String[] fileName = new String[nData];
            System.out.println("classGender = " + classGender[g]);
            for (int i = 0; i < nData; i++) {
                c = new SPM_Centrist(2);
                c.extract(folderGenderTraining.listFiles()[i].toString());
                fileName[i] = folderGenderTraining.listFiles()[i].getName();
                Mat mat = Imgcodecs.imread(folderGenderTraining.listFiles()[i].toString(), Imgcodecs.IMREAD_GRAYSCALE);
                if (g == 0) {
                    dataMale.add(c.getHistogram());
                } else {
                    dataFemale.add(c.getHistogram());
                }
                n++;
            }

            if (g == 0) {
                CsvUtils.writeAListDoubleToCSV(dataMale, PATH_TRAINING + "testManual_male.csv");
                CsvUtils.writeStringToCSV(fileName, PATH_TRAINING + "testManualName_male.csv");
            } else {
                CsvUtils.writeAListDoubleToCSV(dataFemale, PATH_TRAINING + "testManual_female.csv");
                CsvUtils.writeStringToCSV(fileName, PATH_TRAINING + "testManualName_female.csv");
            }
        }
    }

    public static void getDataFromCSV() throws FileNotFoundException {
        dataTestMale = CsvUtils.getAListDataFromText(PATH_TRAINING + "testManual_male.csv", nData, totFeatures);
        dataTestFemale = CsvUtils.getAListDataFromText(PATH_TRAINING + "testManual_female.csv", nData, totFeatures);
    }

    public static void testPCA() throws UnsupportedEncodingException, IOException {
        ArrayList<String[]> dataTest = new ArrayList<>();
        PCA pca;
        String[] weightTest;
        for (int i = 0; i < dataTestMale.size(); i++) {
//            dataTest.add(pca.testing(dataTestMale.get(i), classGender[0]));
            weightTest = new String[(block * pcaK) + 1];
            for (int j = 0; j < block; j++) {
                System.out.println("======================================================");
                System.out.println("Male " + (i + 1) + " - Block : " + (j + 1));
                double[] tmp = new double[pcaFeatures];
                tmp = Normalization.getChunkArray(dataTestMale.get(i), pcaFeatures, j);

                pca = new PCA(pcaK, pcaFeatures);
                String[] weight = pca.test(tmp, classGender[0], j);
                System.arraycopy(weight, 0, weightTest, j * pcaK, weight.length);
                weightTest[weightTest.length - 1] = classGender[0];
            }
            dataTest.add(weightTest);
        }
        for (int i = 0; i < dataTestFemale.size(); i++) {
//            dataTest.add(pca.testing(dataTestFemale.get(j), classGender[1]));
            weightTest = new String[(block * pcaK) + 1];
            for (int j = 0; j < block; j++) {
                System.out.println("======================================================");
                System.out.println("Female " + (i + 1) + " - Block : " + (j + 1));
                double[] tmp = new double[pcaFeatures];
                tmp = Normalization.getChunkArray(dataTestFemale.get(i), pcaFeatures, j);

                pca = new PCA(pcaK, pcaFeatures);
                String[] weight = pca.test(tmp, classGender[1], j);
                System.arraycopy(weight, 0, weightTest, j * pcaK, weight.length);
                weightTest[weightTest.length - 1] = classGender[1];
            }
            dataTest.add(weightTest);
        }

        CsvUtils.writeToCSVwithLabel(dataTest, PATH_TRAINING + "pca_test_manual.csv");
    }

    public static void testSVM() throws Exception {
        ConverterUtils.DataSource src = null;
        Instances data_train = null;
        Instances data_test = null;
        Evaluation eva = null;
        double[] prediction = null;
        try {
            src = new ConverterUtils.DataSource(PATH_TRAINING_BHEP + "centristBHEP_pca_" + pcaK + "_train.csv");
            data_train = src.getDataSet();
            data_train.setClass(data_train.attribute("class"));
            SMO smo = new SMO();

            //train and build classifier
            RBFKernel rbf = new RBFKernel();
            rbf.setGamma(sigma);
            smo.setKernel(rbf);
            smo.buildClassifier(data_train);

            src = new ConverterUtils.DataSource(PATH_TRAINING + "pca_test_manualBHEP.arff");
            data_test = src.getDataSet();
            data_test.setClass(data_train.attribute("class"));
            //test
            eva = new Evaluation(data_train);
            prediction = eva.evaluateModel(smo, data_test);
        } catch (Exception ex) {
            Logger.getLogger(GUI.class.getName()).log(Level.SEVERE, null, ex);
        }

        for (int i = 0; i < eva.numInstances(); i++) {
            double realVal = data_test.instance(i).classValue();
            if (realVal != prediction[i]) {
                System.out.print("- Data " + (i + 1) + " " + data_test.instance(i).stringValue(data_test.attribute("class")));
                System.out.println("("+realVal+") - Prediction ("+prediction[i]+") : False");
            }
        }
        System.out.println("-------------------------------------------------------------------------------------");
        System.out.println("- Accuracy : " + eva.pctCorrect() + "%");
        System.out.println("- Misclassification Rate : " + eva.pctIncorrect() + "%");
        System.out.println(eva.toMatrixString(""));
    }

    public static void main(String[] args) throws FileNotFoundException, IOException, Exception {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        centrist();
//        getDataFromCSV();
//        testPCA();
        testSVM();
    }
}
