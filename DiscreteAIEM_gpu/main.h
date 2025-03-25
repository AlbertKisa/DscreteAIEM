#ifndef MAIN_HEADER
#define MAIN_HEADER

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <complex>
#include <ctime>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <device_launch_parameters.h>


using std::complex;
using std::cout;
using std::endl;


#define  PI			3.1415926535897932384626433832795
#define  VC			2.99792458e8 
#define  nodetime   200



void SSA1_linear(complex<double> eps_r, double vk, double the_s, double phi_s,
    double the_i, double phi_i, double UU, double t, complex<double>* b);
void SSA1_linear_cuda(cuDoubleComplex eps_r, double vk, double the_s, double phi_s,
    double the_i, double phi_i, double UU, double t, cuDoubleComplex* b);
void Wpm(double U, double kx, double ky, double& skf);
void omega(double kx, double ky, double& omg);
void Intk(int Num, double delt, complex<double>** Matrix_int, complex<double>& Sum);
void IFT_mathematica(int Num, complex<double>* Matrix_ori, complex<double>* ISAR1_t1);
void IFT_plus(int Num, complex<double>* Matrix_int, complex<double>* Matrix_out);
void Ints(int Num, double delt, complex<double>** Matrix_int, complex<double>& Sum);
void dot_product_1(double* a, double* b, int size, double& n);

#endif // !MAIN_HEADER

