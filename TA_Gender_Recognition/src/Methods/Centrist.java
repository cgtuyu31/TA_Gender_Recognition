/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Methods;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import net.semanticmetadata.lire.utils.ImageUtils;
import org.opencv.core.Core;
import static org.opencv.core.CvType.CV_8U;
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
public class Centrist {

    double histogram[];
    Mat img_ct;
    static int i = 0;
    static int p = 0;

    public Centrist() {
        img_ct = new Mat();
    }

    public void extract(Mat img) {
        p = 0;
        img_ct.create(img.size(), CV_8U);

        histogram = new double[256];
        for (int i = 0; i < histogram.length; i++) {
            histogram[i] = 0;
        }

        // BGR kalo ga di ubah jd grayscale
        // x = baris, y = kolom
        for (int x = 1; x < (int) img.size().width - 1; x++) {
            for (int y = 1; y < (int) img.size().height - 1; y++) {
                int ctValue = 0;
                double intensity = img.get(x, y)[0];

                //    [x-1, y-1] [x-1, y] [x-1, y+1] | 128 64 32
                //    [x, y-1]            [x, y+1]   |  16     8
                //    [x+1, y-1] [x+1, y] [x+1, y+1] |   4  2  1
                if (img.get(x + 1, y + 1)[0] <= intensity) {
                    ctValue += 1;
                }
                if (img.get(x + 1, y)[0] <= intensity) {
                    ctValue += 2;
                }
                if (img.get(x + 1, y - 1)[0] <= intensity) {
                    ctValue += 4;
                }
                if (img.get(x, y + 1)[0] <= intensity) {
                    ctValue += 8;
                }
                if (img.get(x, y - 1)[0] <= intensity) {
                    ctValue += 16;
                }
                if (img.get(x - 1, y + 1)[0] <= intensity) {
                    ctValue += 32;
                }
                if (img.get(x - 1, y)[0] <= intensity) {
                    ctValue += 64;
                }
                if (img.get(x - 1, y - 1)[0] <= intensity) {
                    ctValue += 128;
                }
                histogram[Math.min(ctValue, 255)]++;
                img_ct.put(x, y, ctValue);
                p++;
            }
        }
        Imgcodecs.imwrite("G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Dataset\\ct_male\\" + i + "_ct.jpg", img_ct);
    }

    public double[] getHistogram() {
        return histogram;
    }
    
    public static int getP(){
        return p;
    }

    ///test
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        final File folder = new File("G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Dataset\\crop_lfw_male");
        //    String path = "G:\\Glenn\\Kuliah\\Bahan TA\\Java Projects\\TA_Dataset\\crop_lfw_male\\0_face.jpg";

        for (final File fileEntry : folder.listFiles()) {
            Centrist c = new Centrist();
            Mat img = Imgcodecs.imread(fileEntry.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
            c.extract(img);
            System.out.println("Done!!!");
            i++;
        }
    }
}
