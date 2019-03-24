/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Methods;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import javax.swing.text.html.HTML;
import net.semanticmetadata.lire.utils.ImageUtils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import static org.opencv.core.CvType.CV_8S;
import static org.opencv.core.CvType.CV_8U;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
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
public class SPM_Centrist {

    static final int histLength = 256;
    static int block = 0;
    static int level = 2;
    static double[] histogram;
    
    public SPM_Centrist() {
        int block = 0;
        for (int i = 0; i <= level; i++) {
            block += Math.pow(2, (2 * i)) + Math.pow((Math.pow(2, i) - 1), 2);
        }
        histogram = new double[histLength * block];
    }

    public SPM_Centrist(int level) {
        int block = 0;
        for (int i = 0; i <= level; i++) {
            block += Math.pow(2, (2 * i)) + Math.pow((Math.pow(2, i) - 1), 2);
        }
        histogram = new double[histLength * block];
    }

    private static Mat preprocessing(String imagePath) {
//        //get img -> convert to YUV
//        Mat img_ori = Imgcodecs.imread(imagePath, Imgcodecs.IMREAD_UNCHANGED);
//        Mat img_yuv = new Mat();
//        Imgproc.cvtColor(img_ori, img_yuv, Imgproc.COLOR_RGB2YCrCb);
//
//        // get the Y channel
////        Mat img_y = Imgcodecs.imread(imagePath, Imgcodecs.IMREAD_GRAYSCALE);
//        Mat img_y = new Mat();
//        img_y.create(img_yuv.size(), CV_8U);
//        for (int i = 0; i < img_yuv.rows(); i++) {
//            for (int j = 0; j < img_yuv.cols(); j++) {
//                img_y.put(i, j, img_yuv.get(i, j)[0]);
//            }
//        }
//
//        Mat img_y_sharpened = new Mat();
//        img_y_sharpened.create(img_yuv.size(), CV_8U);
//
//        //sharpen image
//        Imgproc.GaussianBlur(img_y, img_y_sharpened, new Size(0, 0), 3);
//        Core.addWeighted(img_y, 1.5, img_y_sharpened, -0.5, 0, img_y_sharpened);
//
//        //calculate histogram
//        double[] histH = createHist(img_y_sharpened);
//        double[] test = new double[256];
////MASIH SALAH INI BAGINYA GA RATA
//        //divide histogram using Harmonic Mean
//        int hm = harmonicMean(img_y_sharpened);
//
//        //apply HE on each histogram
////        Mat img_hu = new Mat(img_y_sharpened.rows(), img_y_sharpened.cols(), img_y_sharpened.type());
//        Imgproc.equalizeHist(histH, test);
//        //concatenate histogram
//        //smooth image
//        //concatenate Y to UV
//        //convert YUV to RGB
        Mat img_ori = Imgcodecs.imread(imagePath, Imgcodecs.IMREAD_GRAYSCALE);
        return img_ori;
    }

    static void extract(String imagePath) {
        Mat img = preprocessing(imagePath);
        Mat cropImg;
        Mat resizedImg = new Mat();
        Size sz = new Size(250, 250);

        if (level >= 0) {

        }
        // level 0:
        Centrist centrist = new Centrist();
        centrist.extract(img);
        System.arraycopy(centrist.getHistogram(), 0, histogram, 0, histLength);

        // level 1:
        int w = img.cols() / 2;
        int h = img.rows() / 2;

//        Rect crop = new Rect(1 * wstep, 1 * hstep, wstep, hstep);
//        Mat cropImg = new Mat(oriImg, crop);
//        Mat resizeImg = new Mat();
//        Size sz = new Size(250, 250);
//        Imgproc.resize(cropImg, resizeImg, sz);
        Rect rectCrop = new Rect(0, 0, w, h);
        //--- resize
        cropImg = new Mat(img, rectCrop);
        Imgproc.resize(cropImg, resizedImg, sz);
        //resize ---
        centrist.extract(resizedImg);
        System.arraycopy(centrist.getHistogram(), 0, histogram, histLength * 1, histLength);

        rectCrop = new Rect(w, 0, w, h);
        //--- resize
        cropImg = new Mat(img, rectCrop);
        Imgproc.resize(cropImg, resizedImg, sz);
        //resize ---
        centrist.extract(resizedImg);
        System.arraycopy(centrist.getHistogram(), 0, histogram, histLength * 2, histLength);

        rectCrop = new Rect(0, h, w, h);
        //--- resize
        cropImg = new Mat(img, rectCrop);
        Imgproc.resize(cropImg, resizedImg, sz);
        //resize ---
        centrist.extract(resizedImg);
        System.arraycopy(centrist.getHistogram(), 0, histogram, histLength * 3, histLength);

        rectCrop = new Rect(w, h, w, h);
        //--- resize
        cropImg = new Mat(img, rectCrop);
        Imgproc.resize(cropImg, resizedImg, sz);
        //resize ---
        centrist.extract(resizedImg);
        System.arraycopy(centrist.getHistogram(), 0, histogram, histLength * 4, histLength);

        // and that's the additional sub image in level one:
        rectCrop = new Rect(w / 2, h / 2, w, h);
        //--- resize
        cropImg = new Mat(img, rectCrop);
        Imgproc.resize(cropImg, resizedImg, sz);
        //resize ---
        centrist.extract(resizedImg);
        System.arraycopy(centrist.getHistogram(), 0, histogram, histLength * 5, histLength);

        // level 2:
        int wstep = img.cols() / 4;
        int hstep = img.rows() / 4;
        int binPos = 6; // the next free section in the histogram
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                rectCrop = new Rect(i * wstep, j * hstep, wstep, hstep);
                //--- resize
                cropImg = new Mat(img, rectCrop);
                Imgproc.resize(cropImg, resizedImg, sz);
                //resize ---
                centrist.extract(resizedImg);
                System.arraycopy(centrist.getHistogram(), 0, histogram, histLength * binPos, histLength);
                binPos++;
            }
        }
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                rectCrop = new Rect(wstep / 2 + i * wstep, hstep / 2 + j * hstep, wstep, hstep);
                //--- resize
                cropImg = new Mat(img, rectCrop);
                Imgproc.resize(cropImg, resizedImg, sz);
                //resize ---
                centrist.extract(resizedImg);
                System.arraycopy(centrist.getHistogram(), 0, histogram, histLength * binPos, histLength);
                binPos++;
            }
        }
        System.out.println("Extract CENTRIST "+imagePath+" DONE!!");
    }

    static double[] getHistogram() {
        return histogram;
    }

    private static int harmonicMean(Mat img) {
        double n = img.rows() * img.cols();

        double total = 0;
        for (int i = 0; i < img.rows(); i++) {
            for (int j = 0; j < img.cols(); j++) {
                double inten = img.get(i, j)[0];
                if (inten != 0) {
                    total += 1 / inten;
                }
            }
        }
        System.out.println("n = " + n);
        System.out.println("total = " + total);

        double hm = n / total;

        return (int) hm;
    }

    private static double[] createHist(Mat img) {
        double[] hist = new double[256];
        for (int i = 0; i < hist.length; i++) {
            hist[i] = 0;
        }
        for (int i = 0; i < img.rows(); i++) {
            for (int j = 0; j < img.cols(); j++) {
                hist[(int) img.get(i, j)[0]]++;
            }
        }
        return hist;
    }

    ///test
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        SPM_Centrist c = new SPM_Centrist(2);

        String path = "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Dataset\\0_face.jpg";

        String[] pathCropGenderTrain = {
            "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Dataset\\crop_lfw_male",
            "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Dataset\\crop_lfw_female"};
        Mat img_ori = c.preprocessing(path);
        Imgcodecs.imwrite("G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Dataset\\img_y.jpg", img_ori);

        int count = 0;
        System.out.println("path: "+path);
        c.extract(path);
        for (int i = 0;
                i < c.getHistogram().length; i++) {
            if (c.getHistogram()[i] == 0) {
                count++;
            } else {
                System.out.println(i+". "+c.getHistogram()[i]);
            }
        }

//        System.out.println("->> " + count);
//        Mat img = Imgcodecs.imread(path, Imgcodecs.IMREAD_GRAYSCALE);
//        System.out.println("rows : " + img.rows() + " - cols : " + img.cols());
//
//        Rect rectCrop = new Rect(0, 0, img.cols() / 2, img.rows() / 2);
//        Mat crophalf = new Mat(img, rectCrop);
//        System.out.println("rows : " + crophalf.rows() + " - cols : " + crophalf.cols());
//
//        rectCrop = new Rect(0, 0, img.cols() / 4, img.rows() / 4);
//        Mat cropquart = new Mat(img, rectCrop);
//        System.out.println("rows : " + cropquart.rows() + " - cols : " + cropquart.cols());
//        Mat oriImg = Imgcodecs.imread(path, Imgcodecs.IMREAD_GRAYSCALE);
//        int w = oriImg.cols() / 2;
//        int h = oriImg.rows() / 2;
//        int wstep = oriImg.cols() / 4;
//        int hstep = oriImg.rows() / 4;
//
//        Rect crop = new Rect(1 * wstep, 1 * hstep, wstep, hstep);
//        Mat cropImg = new Mat(oriImg, crop);
//        Mat resizeImg = new Mat();
//        Size sz = new Size(250, 250);
//        Imgproc.resize(cropImg, resizeImg, sz);
//        Imgcodecs.imwrite("G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Dataset\\1_resizeface.jpg", resizeImg);
//        crop = new Rect(0 * wstep, 1 * hstep, wstep, hstep);
//        cropImg = new Mat(oriImg, crop);
//        Imgproc.resize(cropImg, resizeImg, sz);
//        Imgcodecs.imwrite("G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Dataset\\2_resizeface.jpg", resizeImg);
    }
}
