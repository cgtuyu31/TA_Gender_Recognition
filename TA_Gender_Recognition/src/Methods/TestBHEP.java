/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Methods;

import java.awt.image.BufferedImage;
import java.awt.image.RescaleOp;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import javax.imageio.ImageIO;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import ta_gender_recognition.GUI;

/**
 *
 * @author Tuyu
 */
public class TestBHEP {

    public static void main(String[] args) throws IOException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        String[] pathCropGenderTrain = {
//            GUI.PATH_HEADER_DATASET + "manual_crop_male",
//            GUI.PATH_HEADER_DATASET + "manual_crop_female"};
//        String[] pathLowerGenderTrain = {
//            GUI.PATH_HEADER_DATASET + "manual_lower_male",
//            GUI.PATH_HEADER_DATASET + "manual_lower_female"};
//        for (int g = 0; g < GUI.classGender.length; g++) {
//            File folderGenderTraining = new File(pathCropGenderTrain[g]);
//            File[] files = folderGenderTraining.listFiles();
//            Arrays.sort(folderGenderTraining.listFiles(), (f1, f2) -> f1.compareTo(f2));
//            for (int i = 0; i < files.length; i++) {
//                System.out.println((i + 1) + ". : " + files[i].toString());
//                BufferedImage img = ImageIO.read(folderGenderTraining.listFiles()[i]);
//                RescaleOp rescaleOp = new RescaleOp(0.7f, 10, null);
//                rescaleOp.filter(img, img);
//                File f1 = new File(pathLowerGenderTrain[g] + "\\lower_" + i + ".jpg");
//                ImageIO.write(img, "jpg", f1);
//            }
//        }

//        PreprocessingBHEP bhep = new PreprocessingBHEP();
//        Mat img = bhep.getBHEP(GUI.PATH_HEADER_DATASET + "manual_crop_female\\face_0.jpg");
//        Imgproc.cvtColor(img, img, Imgproc.COLOR_RGB2GRAY);
//        Imgcodecs.imwrite(GUI.PATH_HEADER_DATASET + "manual_crop_female\\zbhep_z6.jpg", img);
//        img = Imgcodecs.imread(GUI.PATH_HEADER_DATASET + "manual_crop_female\\face_0.jpg", Imgcodecs.IMREAD_GRAYSCALE);
//        Imgcodecs.imwrite(GUI.PATH_HEADER_DATASET + "manual_crop_female\\zgray_z6.jpg", img);
//        img = Imgcodecs.imread(GUI.PATH_HEADER_DATASET + "manual_crop_male\\lower_27.jpg", Imgcodecs.IMREAD_GRAYSCALE);
//        Imgproc.equalizeHist(img, img);
//        Imgcodecs.imwrite(GUI.PATH_HEADER_DATASET + "manual_crop_male\\zhe_lower2.jpg", img);
        
//        img = bhep.getBHEP(GUI.PATH_HEADER_DATASET + "manual_crop_male\\10_face.jpg");
//        Imgproc.cvtColor(img, img, Imgproc.COLOR_RGB2GRAY);
//        Imgcodecs.imwrite(GUI.PATH_HEADER_DATASET + "manual_crop_male\\audric_gray_10.jpg", img);
    }
}
