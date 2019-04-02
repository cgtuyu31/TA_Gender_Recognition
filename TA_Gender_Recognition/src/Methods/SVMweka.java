/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Methods;

import java.io.File;
import java.util.ArrayList;
import static org.opencv.ml.SVM.RBF;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author Tuyu
 */
public class SVMweka {

    private SMO model;
    private Instances data;

    /**
     * Returns a LibSVM model object that was loaded or trained.
     *
     * @return LibSVM model object that was loaded or trained.
     */
    public SMO getModel() {
        return this.model;
    }

    /**
     * Sets a LibSVM model object.
     *
     * @return void
     */
    public void setModel(SMO model) {
        this.model = model;
    }

    /**
     * Returns a set of Instances created by loading a CSV file.
     *
     * @return Instances Instances data representing a CSV file.
     */
    public Instances getInstances() {
        return this.data;
    }

    /**
     * Set Instances
     *
     */
    public void setInstances(Instances instances) {

        this.data = instances;
    }

    /**
     * This method loads the CSV file to Instances
     *
     * @param filename The destination to a CSV file
     * @return void
     */
    public void loadCSV(String filename) {
        try {
//            ConverterUtils.DataSource source = new ConverterUtils.DataSource(filename);
//            Instances data = source.getDataSet();
            CSVLoader trainingLoader = new CSVLoader();
            trainingLoader.setSource(new File(filename));
            trainingLoader.setNoHeaderRowPresent(false);
            data = trainingLoader.getDataSet();
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        
//CENTRIST COBAIN LIBRARY
//SMO LIAT DI PERPUS PNY ALUMNI
    }

    /**
     * This method trains and builds a model based on a loaded CSV file. Please
     * note that it is important to include all proper parameters, and to do
     * that, please use WEKA explorer GUI
     *
     * @param options The SVM training options (or parameters).
     * @return void
     */
    public void buildModel() {
        try {
            model = new SMO();
            model.setOptions(weka.core.Utils.splitOptions("-C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K "
                + "\"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.01\""));
//            RBFKernel rbf = new RBFKernel();
//            rbf.setGamma(g);
//            model.setKernel(rbf);
            model.buildClassifier(this.data);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * This method saves the trained model to a file
     *
     * @param filename Desired destination to save the SVM model
     * @return void
     */
    public void saveModelToFile(String filename) {
        try {
            weka.core.SerializationHelper.write(filename, model);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * This method loads the trained model from a a saved file Please note that
     * it is important to include all proper parameters, and to do that, please
     * use WEKA explorer GUI
     *
     * @param filename Desired destination to load the SVM model from
     * @return void
     */
    public void loadModelFromFile(String filename) {
        try {
            model = (SMO) weka.core.SerializationHelper.read(filename);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * This method is made to quickly predict just one row of input data Please
     * note that this method accepts only numeric values, while the last value
     * can be anything (but must be included in a row), since it is the value to
     * be predicted
     *
     * @param row The delimited row of values to make a prediction from
     * @param delimiters The delimiters used to delimit the values in a row (for
     * example, comma, tab, space,...)
     * @return double the prediction value
     */
    public double predictForRow(String row, String delimiters) {
        try {

            String[] split = row.split(delimiters);
            int sz = split.length - 1;
            double[] raw = new double[split.length];
            for (int t = 0; t < sz; t++) {
                raw[t] = Double.parseDouble(split[t]);
            }

            ArrayList<Attribute> atts = new ArrayList<Attribute>(sz);

            for (int t = 0; t < sz + 1; t++) {
                atts.add(new Attribute("name" + t, t));
            }

            Instances x = new Instances(data);
            Instances dataRaw = new Instances("TestInstances", atts, sz);
            dataRaw.add(new DenseInstance(1.0, raw));
            Instance first = dataRaw.firstInstance(); //System.out.println(first);
            int cIdx = dataRaw.numAttributes() - 1;
            dataRaw.setClassIndex(cIdx);

            return model.classifyInstance(first);

        } catch (Exception e) {
            e.printStackTrace();
        }
        return 0;

    }

    /**
     * This method is made to classify a single instance.
     *
     * @param instance The instance to classify
     * @return double the prediction value
     */
    public double classifyInstance(Instance instance) {

        double ret = 0;
        try {
            ret = model.classifyInstance(instance);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return ret;
    }
}
