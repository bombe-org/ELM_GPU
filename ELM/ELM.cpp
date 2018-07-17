// author : Liang Li
// email : liliang@stumail.neu.edu.cn

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <helper_timer.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cula.h>
#include <cula_blas_device.h>
#define  N_COUNT 581012
#define IN_COUNT 54
#define L_COUNT 100
#define C_COUNT 0.1

#define GPU
#include "Eigen/Core"
#include "Eigen/Cholesky"

using namespace std;
using namespace Eigen;

int compare(const void *a, const void *b);
MatrixXd buildTargetMatrix(double *Y, int nLabels);


template<typename Derived>
int elmTrain(double *X, int dims, int nsmp,
	double *Y,
	const int nhn, const double C,
	MatrixBase<Derived> &inW, MatrixBase<Derived> &bias, MatrixBase<Derived> &outW) {

	MatrixXd mX = Map<MatrixXd>(X, dims, nsmp);
	MatrixXd mTargets = buildTargetMatrix(Y, nsmp);
	cout << "loading data finished\n";
	inW = MatrixXd::Random(nhn, dims);
	bias = MatrixXd::Random(nhn, 1);
	MatrixXd preH = inW * mX + bias.replicate(1, nsmp);
	MatrixXd H = (1 + (-preH.array()).exp()).cwiseInverse();
	cout << "preparing data finished\n";
#ifdef CPU
	clock_t t1,t2;
	t1 = clock();
	// build matrices to solve Ax = b
	MatrixXd A = (MatrixXd::Identity(nhn, nhn)).array() * (1 / C) + (H * H.transpose()).array();      //nhn*nhn
	t2 = clock();
	cout<< "computing A with CPU:" << (double)(t2 - t1)/CLOCKS_PER_SEC <<"s\n";
	t1 = clock();
	MatrixXd b = H * mTargets.transpose();  //    nhn*7
	t2 = clock();
	cout<< "computing b with CPU:" << (double)(t2 - t1)/CLOCKS_PER_SEC <<"s\n";
	// solve the output weights as a solution to a system of linear equations
	t1 = clock();
	outW = A.llt().solve(b);
	t2 = clock();
	cout<< "solving with CPU:" << (double)(t2 - t1)/CLOCKS_PER_SEC <<"s\n";
#endif

#ifdef GPU
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

	StopWatchInterface *timer;  
	cudaThreadSynchronize();
	sdkCreateTimer(&timer);  
	sdkStartTimer(&timer);  
	cudaMalloc((void **)&device_H, nsmp*nhn * sizeof(float));
	cudaMalloc((void **)&device_H1, nsmp*nhn * sizeof(float));
	cudaMalloc((void **)&device_T, 7*nhn * sizeof(float));
	cudaMalloc((void **)&device_A, nhn*nhn * sizeof(float));
	cudaMalloc((void **)&device_b, nhn*7 * sizeof(float));
	cudaMalloc((void **)&device_outw, nhn*7 * sizeof(float));
	cudaMemcpy(host_H,device_H, nsmp*nhn*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(host_H1,device_H1, nsmp*nhn*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(host_T,device_T, nsmp*nhn*sizeof(float),cudaMemcpyHostToDevice);  
	cudaThreadSynchronize();
	sdkStopTimer(&timer);  
	double dSeconds = sdkGetTimerValue(&timer);  
	cout<<"offloading cost:"<<dSeconds<<"ms\n";

	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
	culaDeviceSgemm('N','N',nhn,nhn,nsmp,1,device_H1,nhn,device_H,nsmp,0,device_A,nhn);
	cudaThreadSynchronize();
	sdkStopTimer(&timer);  
	dSeconds = sdkGetTimerValue(&timer);  
	cout<<"computing A:"<<dSeconds<<"ms\n";

	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
	culaDeviceSgemm('N','N',7,nhn,nsmp,1,device_T,7,device_H,nsmp,0,device_b,7);
	cudaThreadSynchronize();
	sdkStopTimer(&timer);  
	dSeconds = sdkGetTimerValue(&timer);  
	cout<<"computing b:"<<dSeconds<<"ms\n";


	double ipiv = 0.1;	
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
	culaDeviceSgemm('N','N',7,nhn,nhn,1,device_b,7,device_A,nhn,0,device_outw,7);
	cudaThreadSynchronize();
	sdkStopTimer(&timer);  
	dSeconds = sdkGetTimerValue(&timer);  
	cout<<"solving:"<<dSeconds<<"ms\n";

	float *host_outW = (float*)malloc(sizeof(float)*7*nhn);
	cudaMemcpy(host_outW,device_outw, nhn*7*sizeof(float),cudaMemcpyDeviceToHost);
	outW = Map<MatrixXd>((double*)host_outW,nhn,7);
#endif
	return 0;
}

template<typename Derived>
int elmPredict(double *X, int dims, int nsmp,
	MatrixBase<Derived> &mScores,
	MatrixBase<Derived> &inW, MatrixBase<Derived> &bias, MatrixBase<Derived> &outW) {

	MatrixXd mX = Map<MatrixXd>(X, dims, nsmp);
	MatrixXd preH = inW * mX + bias.replicate(1, nsmp);
	MatrixXd H = (1 + (-preH.array()).exp()).cwiseInverse();
	mScores = (H.transpose() * outW).transpose();
	return 0;
}

int compare(const void *a, const void *b) {
	const double *da = (const double *) a;
	const double *db = (const double *) b;
	return (*da > *db) - (*da < *db);
}

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
	Eigen::setNbThreads(2);
	cout<<Eigen::nbThreads()<<"\n";
	double *x = (double *) malloc(IN_COUNT * N_COUNT * sizeof(double));
	double *y = (double *) malloc(N_COUNT * sizeof(double));

	std::ifstream fin("..\\covtype.data");
	std::string line;
	long long int row = 0;
	int column;
	double data[55];
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
	cout<< "training finished\n";
	//elmPredict(x, IN_COUNT, N_COUNT, mScore, inW, bias, outW);
	return 0;
}
