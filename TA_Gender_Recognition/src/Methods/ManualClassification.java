/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Methods;

import Jama.Matrix;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Scanner;

/**
 *
 * @author Tuyu
 */
public class ManualClassification {
    
     private double calculateRBFKernel(double[][] data, double sigma,
            int classSource, int classTarget) {
        double value = 0;
        //data[0].length --> banyaknya fitur untuk 1 data
        for (int i = 0; i < data[0].length; i++) {
            value += Math.pow(data[classSource][i] - data[classTarget][i], 2);
        }
        return Math.exp(-(value) / (2 * Math.pow(sigma, 2)));
    }

    public double[][] createRBFMatrix(double data[][], double sigma) {
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
        /*
            matriks kiri    matriks kanan
            4x  - y  +  z  =    7
            4x  - 8y +  z  =    -21
            -2x +  y + 5x  =    15
        
        Solusi x = 2, y = 4, z = 3
         */

        //matriks persamaan linear (kiri)
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
}
