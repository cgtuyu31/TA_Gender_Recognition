/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Methods;

import java.io.File;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

/**
 *
 * @author Tuyu
 */
public class FaceDetector {

    private static CascadeClassifier faceCascade;

    public Mat[] snipFace(String image, Size size) {
        Mat matImage = Imgcodecs.imread(image, Imgcodecs.IMREAD_UNCHANGED);
        Rect[] rectFace = detectFace(matImage);
        int rectFaceLength = rectFace.length;

        Mat[] matFace = new Mat[rectFaceLength];

        for (int i = 0; i < rectFaceLength; i++) {

            matFace[i] = matImage.submat(rectFace[i]);
            Imgproc.resize(matFace[i], matFace[i], size);

            //Highgui.imwrite(image.substring(0, image.length()-4)+"Snipped"+i+image.substring(image.length()-4), matFace[i]);
        }

        return matFace;
    }

    private Rect[] detectFace(Mat matImage) {
        MatOfRect faces = new MatOfRect();
        String HumanFace = "src/res/knowledge/haarcascade_frontalface_alt.xml";

        CascadeClassifier cascadeClassifier = new CascadeClassifier(HumanFace);

        cascadeClassifier.detectMultiScale(matImage, faces, 1.1, 10, Objdetect.CASCADE_DO_CANNY_PRUNING, new Size(20, 20),
                matImage.size());

        //System.out.println(faces.dump());///test
        return faces.toArray();
    }

    ///test
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

//      ------------------------------------------------------------------
        final File folder = new File("G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Dataset\\lfw_male_test");

        listFile(folder);
    }

//    public static void initHaarCascade() {
//        String url = "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\GenderRecognizer-master\\src\\res\\knowledge";
//
//        faceCascade = new CascadeClassifier(url);
//
//        if (faceCascade.load("src/res/haarcascade_frontalface_alt.xml")) {
//            System.out.println("Successfully Load Face Cascade");
//        } else {
//            System.out.println("Failed Load Face Cascade");
//        }
//    }

    public static void listFile(final File folder) {
        int i = 0;
        for (final File fileEntry : folder.listFiles()) {
            if (fileEntry.isDirectory()) {
                listFile(fileEntry);
                new File(fileEntry.getPath()).mkdir();

                System.out.println(fileEntry.getName());
            } else {
                System.out.println(fileEntry.getAbsolutePath());
                FaceDetector faceDetector = new FaceDetector();
                Mat[] mats = faceDetector.snipFace(fileEntry.getAbsolutePath(), new Size(250, 250));

                for (Mat mat : mats) {
//            Imgcodecs.imwrite(imagePath.substring(0, imagePath.length() - 4) + "Snipped" + i + imagePath.substring(imagePath.length() - 4),
//                    mat);
                    Imgcodecs.imwrite("G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Dataset\\crop_lfw_male_test\\" + i + "_face.jpg", mat);
                }

                System.out.println("Done!!!");
            }
            i++;
        }
    }
}
