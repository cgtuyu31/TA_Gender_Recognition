/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Methods;

import java.io.File;
import org.opencv.core.Core;
import static org.opencv.core.CvType.CV_8U;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.*;
import java.text.DecimalFormat;
import javax.imageio.ImageIO;
import org.opencv.core.CvType;

/**
 *
 * @author Tuyu
 */
public class PreprocessingBHEP {

//    public static BufferedImage image1;
    public PreprocessingBHEP() {

    }

    public PreprocessingBHEP(String in) throws IOException {
        File imgFile;
        imgFile = new File(in);
//        image1 = getGrayscaleImage(ImageIO.read(imgFile));
//        BufferedImage image2 = equalizeBHEP(image1);
    }

    static Mat getBHEP(String imagePath) {
        //get img -> convert to YUV
        Mat img_ori = Imgcodecs.imread(imagePath, Imgcodecs.IMREAD_UNCHANGED);
        Mat img_yuv = new Mat();
        if (img_ori.channels() == 3) {
            Imgproc.cvtColor(img_ori, img_yuv, Imgproc.COLOR_RGB2YCrCb);
        } else {
            img_yuv = img_ori;
        }

        // get the Y channel
//        Mat img_y = Imgcodecs.imread(imagePath, Imgcodecs.IMREAD_GRAYSCALE);
        Mat img_y = new Mat();
        img_y.create(img_yuv.size(), CV_8U);
        for (int i = 0; i < img_yuv.rows(); i++) {
            for (int j = 0; j < img_yuv.cols(); j++) {
                img_y.put(i, j, img_yuv.get(i, j)[0]);
            }
        }

        Mat img_y_sharpened = new Mat();
        img_y_sharpened.create(img_yuv.size(), CV_8U);

        //sharpen image
        Imgproc.GaussianBlur(img_y, img_y_sharpened, new Size(0, 0), 3);
        Core.addWeighted(img_y, 1.5, img_y_sharpened, -0.5, 0, img_y_sharpened);

        //bhep
        BufferedImage imgBHEP = mat2Img(img_y_sharpened);
        imgBHEP = equalizeBHEP(imgBHEP);

        Mat equalizedImg = img2Mat(imgBHEP);

        //smooth image
        Mat smoothedImg = new Mat();
        Imgproc.GaussianBlur(equalizedImg, smoothedImg, new Size(0, 0), 3);

        Mat finalImg = new Mat();
        if (img_ori.channels() == 3) {
            //concatenate Y to UV
            for (int i = 0; i < img_yuv.rows(); i++) {
                for (int j = 0; j < img_yuv.cols(); j++) {
                    double[] newpixel = new double[3];
                    newpixel[0] = smoothedImg.get(i, j)[0];
                    newpixel[1] = img_yuv.get(i, j)[1];
                    newpixel[2] = img_yuv.get(i, j)[2];
                    img_yuv.put(i, j, newpixel);
                }
            }

            //convert YUV to RGB
            Imgproc.cvtColor(img_yuv, finalImg, Imgproc.COLOR_YCrCb2RGB);
        }else{
            finalImg = smoothedImg;
        }

//        Mat img_ori = Imgcodecs.imread(imagePath, Imgcodecs.IMREAD_GRAYSCALE);
        return finalImg;
    }

    public PreprocessingBHEP(String in, String out, String gr, String bhep) {

        File f1 = new File(in);
//            File f2 = new File(out);
//            File f3 = new File(gr);
//            File f4 = new File(bhep);

//            BufferedImage image1 = getGrayscaleImage(ImageIO.read(f1));
//            ImageIO.write(image1, "jpg", f3);
//            BufferedImage image2 = equalize(image1);
//            ImageIO.write(image2, "jpg", f2);
//            BufferedImage image3 = equalizeBHEP(image1);
//            ImageIO.write(image3, "jpg", f4);
    }

    public BufferedImage getGrayscaleImage(BufferedImage src) {

        BufferedImage gImg = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_BYTE_GRAY);

        WritableRaster wr = src.getRaster();
        WritableRaster gr = gImg.getRaster();

        for (int i = 0; i < wr.getWidth(); i++) {
            for (int j = 0; j < wr.getHeight(); j++) {
                gr.setSample(i, j, 0, wr.getSample(i, j, 0));
            }
        }

        gImg.setData(gr);
        return gImg;

    }

    public BufferedImage equalize(BufferedImage src) {
        BufferedImage nImg = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_BYTE_GRAY);

        WritableRaster wr = src.getRaster();
        WritableRaster er = nImg.getRaster();
        int totpix = wr.getHeight() * wr.getWidth();
        int[] histogram = new int[256];

        for (int x = 0; x < wr.getWidth(); x++) {
            for (int y = 0; y < wr.getHeight(); y++) {
                histogram[wr.getSample(x, y, 0)]++;
            }
        }

        write_histogram(histogram);

        int[] chistogram = new int[256];
        chistogram[0] = histogram[0];
        for (int i = 1; i < 256; i++) {
            chistogram[i] = chistogram[i - 1] + histogram[i];
        }

        float[] arr = new float[256];
        for (int i = 0; i < 256; i++) {
            arr[i] = (float) (chistogram[i] * 255.0 / (float) totpix);
        }

        write_new_histogram(arr);

        for (int x = 0; x < wr.getWidth(); x++) {
            for (int y = 0; y < wr.getHeight(); y++) {
                int nVal = (int) arr[wr.getSample(x, y, 0)];
                er.setSample(x, y, 0, nVal);
            }
        }

        nImg.setData(er);
        return nImg;
    }

    public static BufferedImage equalizeBHEP(BufferedImage src) {
        BufferedImage nImg = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_BYTE_GRAY);

        WritableRaster wr = src.getRaster();
        WritableRaster er = nImg.getRaster();
        int totpix1 = 0;
        int totpix2 = 0;

        int[] histogram = new int[256];

        //frekuensi derajat keabuan
        for (int x = 0; x < wr.getWidth(); x++) {
            for (int y = 0; y < wr.getHeight(); y++) {
                histogram[wr.getSample(x, y, 0)]++;
            }
        }

        //hitung nilai tengah dengan Harmonic Mean
        int hm = harmonicMean(histogram);

        write_histogramBHEPawal(histogram);

        //distribusi kumulatif hist1
        int[] chist1 = new int[hm];
//        System.out.println("chist1 length " + chist1.length);
        chist1[0] = histogram[0];
        for (int i = 1; i < hm; i++) {
            chist1[i] = chist1[i - 1] + histogram[i];
        }
        totpix1 = chist1[hm - 1];
//        System.out.println("chist1 total " + chist1[hm - 1]);

        //distribusi kumulatif hist2
        int[] chist2 = new int[histogram.length - hm];
//        System.out.println("chist2 length " + chist2.length);
        chist2[0] = histogram[hm];
        for (int i = hm + 1; i < 256; i++) {
            chist2[i - hm] = chist2[i - 1 - hm] + histogram[i];
        }
        totpix2 = chist2[256 - hm - 1];
//        System.out.println("chist2 total " + chist2[256 - hm - 1]);

        //hitung nilai keabuan baru dengan persamaan 2.9 untuk setiap histogram
        float[] arr = new float[256];
        for (int i = 0; i < hm; i++) {
            arr[i] = (float) (chist1[i] * (hm - 1) / (float) totpix1);
        }
        for (int i = hm; i < 256; i++) {
            arr[i] = (float) (chist2[i - hm] * (256 - hm - 1) / (float) totpix2);
        }

//        System.arraycopy(centrist.getHistogram(), 0, histogram, histLength * 1, histLength);
        write_new_histogram(arr);

        //ubah dengan nilai keabuan baru
        for (int x = 0; x < wr.getWidth(); x++) {
            for (int y = 0; y < wr.getHeight(); y++) {
                int nVal = (int) arr[wr.getSample(x, y, 0)];
                er.setSample(x, y, 0, nVal);
            }
        }

        nImg.setData(er);
        return nImg;
    }

    public static int harmonicMean(int[] hist) {
        int n = 0;
        double total = 0;
//        System.out.println("histLength : " + hist.length);
        for (int i = 0; i < hist.length; i++) {
            n += hist[i];
            if (hist[i] != 0) {
                for (int j = 0; j < hist[i]; j++) {
                    if (i != 0) {
                        total += ((double) 1) / i;
                    }
                }
            }
        }

//        System.out.println("n = " + n);
//        System.out.println("total = " + total);
        double hm = n / total;
//        System.out.println("hm " + hm);
//        System.out.println("mathround(hm) " + Math.round(hm));
        return (int) Math.round(hm);
    }

    public static void write_histogram(int[] arr) {
        PrintWriter writer = null;
        try {
            writer = new PrintWriter("histogram.txt", "UTF-8");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
        writer.println("color, frequency");
        for (int i = 0; i < 256; i++) {
            writer.println(i + ", " + arr[i]);
        }
        writer.close();
    }

    public static void write_histogramBHEPawal(int[] arr) {
        PrintWriter writer = null;
        try {
            writer = new PrintWriter("histogramBHEP.txt", "UTF-8");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
        writer.println("color, frequency");
        for (int i = 0; i < 256; i++) {
            writer.println(i + ", " + arr[i]);
        }
        writer.close();
    }

    public static void write_new_histogram(float[] arr) {
        PrintWriter writer = null;
        try {
            writer = new PrintWriter("histogram_equalized.txt", "UTF-8");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
        writer.println("color, frequency");
        for (int i = 0; i < 256; i++) {
            writer.println(i + ", " + (int) arr[i]);
        }
        writer.close();
    }

    public static BufferedImage mat2Img(Mat in) {
        BufferedImage out;
        byte[] data = new byte[in.width() * in.height() * (int) in.elemSize()];
        int type;
        in.get(0, 0, data);

        if (in.channels() == 1) {
            type = BufferedImage.TYPE_BYTE_GRAY;
        } else {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }

        out = new BufferedImage(in.width(), in.height(), type);

        out.getRaster().setDataElements(0, 0, in.width(), in.height(), data);
        return out;
    }

    public static Mat img2Mat(BufferedImage in) {
        Mat out;
        byte[] data;
        int r, g, b;

        if (in.getType() == BufferedImage.TYPE_INT_RGB) {
            out = new Mat(in.getHeight(), in.getWidth(), CvType.CV_8UC3);
            data = new byte[in.getWidth() * in.getHeight() * (int) out.elemSize()];
            int[] dataBuff = in.getRGB(0, 0, in.getWidth(), in.getHeight(), null, 0, in.getWidth());
            for (int i = 0; i < dataBuff.length; i++) {
                data[i * 3] = (byte) ((dataBuff[i] >> 16) & 0xFF);
                data[i * 3 + 1] = (byte) ((dataBuff[i] >> 8) & 0xFF);
                data[i * 3 + 2] = (byte) ((dataBuff[i] >> 0) & 0xFF);
            }
        } else {
            out = new Mat(in.getHeight(), in.getWidth(), CvType.CV_8UC1);
            data = new byte[in.getWidth() * in.getHeight() * (int) out.elemSize()];
            int[] dataBuff = in.getRGB(0, 0, in.getWidth(), in.getHeight(), null, 0, in.getWidth());
            for (int i = 0; i < dataBuff.length; i++) {
                r = (byte) ((dataBuff[i] >> 16) & 0xFF);
                g = (byte) ((dataBuff[i] >> 8) & 0xFF);
                b = (byte) ((dataBuff[i] >> 0) & 0xFF);
                data[i] = (byte) ((0.21 * r) + (0.71 * g) + (0.07 * b)); //luminosity
            }
        }
        out.put(0, 0, data);
        return out;
    }

    public static void main(String[] args) {
        PreprocessingBHEP he;
        String path = "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Hasil_Training\\testHE";
        System.out.println("\nHISTOGRAM EQUALIZATION");
        System.out.println("  Generates image with contrast adjustment using image's histogram");
        System.out.println("USAGE:");
        System.out.println("  java: HistogramEqualizationWithHistogram <input_image> <output_image>");
        System.out.println("  java: HistogramEqualizationWithHistogram <input_image>");
        he = new PreprocessingBHEP(path + "\\input.jpg", path + "\\outputHE.jpg", path + "\\inputGray.jpg", path + "\\outputBHEP.jpg");
    }

}
