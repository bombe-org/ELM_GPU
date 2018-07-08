// ELM.cpp : 定义控制台应用程序的入口点。
//

// author : Liang Li
// email : liliang@stumail.neu.edu.cn

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cula.h>
#include <cula_blas_device.h>
#define  N_COUNT 581012
#define IN_COUNT 54
#define L_COUNT 50
#define C_COUNT 0.1

#define GPU
#include "Eigen/Core"
#include "Eigen/Cholesky"

using namespace std;
using namespace Eigen;

int compare(const void *a, const void *b);

//template <typename Derived>
MatrixXd buildTargetMatrix(double *Y, int nLabels);

// entry function to train the ELM model
// INPUT: X, Y, nhn, C
// OUTPUT: inW, bias, outW
template<typename Derived>
int elmTrain(double *X, int dims, int nsmp,
             double *Y,
             const int nhn, const double C,
             MatrixBase<Derived> &inW, MatrixBase<Derived> &bias, MatrixBase<Derived> &outW) {

// map the samples into the matrix object
    MatrixXd mX = Map<MatrixXd>(X, dims, nsmp);
// build target matrix
    MatrixXd mTargets = buildTargetMatrix(Y, nsmp);
// generate random input weight matrix - inW
    inW = MatrixXd::Random(nhn, dims);
// generate random bias vectors
    bias = MatrixXd::Random(nhn, 1);
// compute the pre-H matrix
    MatrixXd preH = inW * mX + bias.replicate(1, nsmp);
// compute hidden neuron output
    MatrixXd H = (1 + (-preH.array()).exp()).cwiseInverse();
#ifdef CPU
	clock_t t1,t2;
	cout<<"cpu computing\n";
	t1 = clock();
	// build matrices to solve Ax = b
    MatrixXd A = (MatrixXd::Identity(nhn, nhn)).array() * (1 / C) + (H * H.transpose()).array();      //nhn*nhn
    MatrixXd b = H * mTargets.transpose();  //    nhn*7
// solve the output weights as a solution to a system of linear equations
    outW = A.llt().solve(b);
	t2 = clock();
	cout<< t2 - t1 <<"\n";
#endif

#ifdef GPU
	cout<<"gpu computing\n";
	culaInitialize();
	float *host_H = (float*)H.data();
	float *host_H1 = (float*)H.transpose().data();
	float *host_T = (float*)mTargets.transpose().data();
	float *device_H;
	float *device_H1;
	float *device_T;
	float *device_A;
	float *device_b;
	float *device_outw;
	clock_t t1,t2;
	t1 = clock();
	cudaMalloc((void **)&device_H, nsmp*nhn * sizeof(float));
	cudaMalloc((void **)&device_H1, nsmp*nhn * sizeof(float));
	cudaMalloc((void **)&device_T, 7*nhn * sizeof(float));
	cudaMalloc((void **)&device_A, nhn*nhn * sizeof(float));
	cudaMalloc((void **)&device_b, nhn*7 * sizeof(float));
	cudaMalloc((void **)&device_outw, nhn*7 * sizeof(float));
	cudaMemcpy(host_H,device_H, nsmp*nhn*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(host_H1,device_H1, nsmp*nhn*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(host_T,device_T, nsmp*nhn*sizeof(float),cudaMemcpyHostToDevice);
	
	culaDeviceSgemm('N','N',nhn,nhn,nsmp,1,device_H1,nhn,device_H,nsmp,0,device_A,nhn);
	culaDeviceSgemm('N','N',7,nhn,nsmp,1,device_T,7,device_H,nsmp,0,device_b,7);
	int ipiv = 0.1;	
	culaDeviceSgetrf(nhn,nhn,device_A,nhn,&ipiv);
	culaDeviceSgetri(nhn,device_A,nhn,&ipiv);
	culaDeviceSgemm('N','N',7,nhn,nhn,1,device_b,7,device_A,nhn,0,device_outw,7);
	cudaDeviceSynchronize();
	t2 = clock();
	float *host_outW = (float*)malloc(sizeof(float)*7*nhn);
	cudaMemcpy(host_outW,device_outw, nhn*7*sizeof(float),cudaMemcpyDeviceToHost);
	
	cout<< t2 - t1 <<"\n";
	outW = Map<MatrixXd>((double*)host_outW,nhn,7);
#endif

    return 0;
}

// entry function to predict class labels using the trained ELM model on test data
// INPUT : X, inW, bias, outW
// OUTPUT : scores
template<typename Derived>
int elmPredict(double *X, int dims, int nsmp,
               MatrixBase<Derived> &mScores,
               MatrixBase<Derived> &inW, MatrixBase<Derived> &bias, MatrixBase<Derived> &outW) {

    // map the sample into the Eigen's matrix object
    MatrixXd mX = Map<MatrixXd>(X, dims, nsmp);

    // build the pre-H matrix
    MatrixXd preH = inW * mX + bias.replicate(1, nsmp);

    // apply the activation function
    MatrixXd H = (1 + (-preH.array()).exp()).cwiseInverse();

    // compute output scores
    mScores = (H.transpose() * outW).transpose();


    return 0;
}


// --------------------------
// Helper functions
// --------------------------

// compares two integer values
//int compare( const void* a, const void *b ) {
//	return ( *(int *) a - *(int *) b );
//}

int compare(const void *a, const void *b) {
    const double *da = (const double *) a;
    const double *db = (const double *) b;
    return (*da > *db) - (*da < *db);
}

// builds 1-of-K target matrix from labels array
//template <typename Derived>
MatrixXd buildTargetMatrix(double *Y, int nLabels) {

    // make a temporary copy of the labels array
    double *tmpY = new double[nLabels];
    double *Y_unique = new double[nLabels];
    for (int i = 0; i < nLabels; i++) {
        tmpY[i] = Y[i];
    }
    // sort the array of labels
    qsort(tmpY, nLabels, sizeof(double), compare);

    // count unique labels
    int nunique = 0;
    Y_unique[0] = tmpY[0];
    for (int i = 0; i < nLabels - 1; i++) {
        if (tmpY[i] != tmpY[i + 1]) {
            nunique++;
            Y_unique[nunique] = tmpY[i + 1];
        }
    }
    nunique++;

    delete[] tmpY;

    MatrixXd targets(nunique, nLabels);
    targets.fill(0);


    // fill in the ones
    for (int i = 0; i < nLabels; i++) {
        int index = 0;
        while (index++ < nunique) {
            if (Y[i] == Y_unique[index]) {
                targets(index, i) = 1;
                break;
            }
        }
    }
    delete[] Y_unique;
    // normalize the targets matrix values (-1/1)
    targets *= 2;
    targets.array() -= 1;
    return targets;
}

int main() {	
    cout<<Eigen::nbThreads()<<"\n";
    double *x = (double *) malloc(IN_COUNT * N_COUNT * sizeof(double));
    double *y = (double *) malloc(N_COUNT * sizeof(double));

    std::ifstream fin("d:\\covtype.data");
    std::string line;

    long long int row = 0;
    int column;
    int data[55];
    while (getline(fin, line)) {
        char cstr[100000];
        strcpy(cstr, line.c_str());
        char *p = strtok(cstr, ",");
        column = 0;
        while (p) {
            data[column] = atoi(p);

            p = strtok(NULL, ",");
            if (column < IN_COUNT)
                x[row * IN_COUNT + column] = data[column];
            else
                y[row] = data[column];
            ++column;
        }
        ++row;
    }

    MatrixXd inW;    // input weight
    MatrixXd bias;   // b
    MatrixXd outW;   // output weight
    MatrixXd mScore;  //predict result
    
    elmTrain(x, IN_COUNT, N_COUNT, y, L_COUNT, C_COUNT, inW, bias, outW);
    elmPredict(x, IN_COUNT, N_COUNT, mScore, inW, bias, outW);
    return 0;
}
