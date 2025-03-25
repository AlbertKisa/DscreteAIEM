#include <stdio.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <complex>
#include <ctime>
#include "mine.h"
#include "complex0.h"
#include "GlobalViariables.h"
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <omp.h>
#include "Fun.h"

//using namespace std;
//  ��������
#define  PI  3.1415926535897932384626433832795
#define  VC  2.99792458e8 
#define CLOCKS_PER_SEC 1000
#define nodetime 2

void IFT_mathematica(int Num, complex<double> *Matrix_ori, complex<double> *ISAR1_t1)
{
	for (int k = 0; k < Num; k++)
	{
		complex<double> sum_1dft = complex<double>(0.0, 0.0);
		for (int i = 0; i < Num; i++)
		{
			complex<double> Phase = complex<double>(0.0, -2 * (i)*PI / Num * k);
			sum_1dft = sum_1dft + Matrix_ori[i] * exp(Phase);
		}
		if (1)
		{
			ISAR1_t1[k] = (1.0 / sqrt(Num))* sum_1dft;

		}

	}
}

void IFT_plus(int Num, complex<double> *Matrix_int, complex<double> *Matrix_out)
{
	int Num1 = 2 * Num + 1;
	complex<double> *Matrix_temp;
	Matrix_temp = (complex<double>*)malloc(sizeof(complex<double>)* Num1);//
	IFT_mathematica(Num1, Matrix_int, Matrix_temp);
	for (int i = 0; i < Num; i++)
	{
		Matrix_out[i] = 1.0 / sqrt(Num1)*exp(complex<double>(0, 2 * PI*Num / Num1*(i + Num + 1)))*Matrix_temp[i + Num + 1];
	}
	for (int j = Num; j < Num1; j++)
	{
		Matrix_out[j] = 1.0 / sqrt(Num1)*exp(complex<double>(0, 2 * PI*Num / Num1*(j + Num + 1)))*Matrix_temp[j - Num];
	}

	free(Matrix_temp);
	Matrix_temp = NULL;

}

void Ints(int Num, double delt, complex<double>**Matrix_int, complex<double> &Sum)
{
	Sum = 0;
	for (int i = 0; i < 2 * Num + 1; i++)
	{
		for (int j = 0; j < 2 * Num + 1; j++)
		{
			Sum = Sum + Matrix_int[i][j] * pow(delt, 2);
		}
	}
}

void Intk(int Num, double delt, complex<double>**Matrix_int, complex<double> &Sum)
{
	Sum = 0;
	for (int i = 0; i < 2 * Num + 1; i++)
	{
		for (int j = 0; j < 2 * Num + 1; j++)
		{
			Sum = Sum + Matrix_int[i][j] * pow(2 * PI / (2 * Num + 1) / delt, 2);
		}
	}
}

void Wpm(double U, double kx, double ky, double &skf)
{
	double a, b, g;
	double Exp, phik;
	double phiw = 22.5 * 180.0 / PI;
	double FF;
	a = 8.1*pow(10, -3);
	b = 0.74;
	g = 9.81;
	Exp = exp(-b*g*g / (kx*kx + ky*ky + pow(10, -6)) / pow(U, 4));
	phik = pow(((kx / sqrt(kx*kx + ky*ky + pow(10, -10))) + 1) / 2, 2);
	skf = a*Exp*phik / pow((kx*kx + ky*ky + pow(10, -10)), 2) / 2;
}


void omega(double kx, double ky, double &omg)
{
	double g, k;
	g = 9.81;
	k = kx*kx + ky*ky;
	omg = sqrt(g*sqrt(k)*(1.0 + k / 363 / 363));
}

void SSA1_linear(complex<double> eps_r, double vk, double the_s, double phi_s, double the_i, double phi_i, double UU, double t, complex<double> *b)
{
	complex<double> cj = complex<double>(0.0, 1.0);
	double q0, q01, q1, vw, vc = 2.99792458e8, qv0, qv1, Pinc, ksk01 = 0, ksk0;
	complex<double> q02, q2, C10, C20, B1vv, B1hh;
	complex<double>  b1v;
	double freq = vk*VC / 2 / PI;
	double qx, qy, qz;
	double k0x, k0y, ksx, ksy, kix, kiy;
	double ww;

	vw = 2 * PI*freq;
	kix = vk*sin(the_i)*cos(phi_i);
	kiy = vk*sin(the_i)*sin(phi_i);
	double ki[2] = { kix, kiy };

	k0x = vk*sin(the_i)*cos(phi_i);
	k0y = vk*sin(the_i)*sin(phi_i);
	ksx = vk*sin(the_s)*cos(phi_s);
	ksy = vk*sin(the_s)*sin(phi_s);
	qx = ksx - k0x;
	qy = ksy - k0y;
	double kspk0[2] = { qx, qy };
	qv0 = vw*vw / vc / vc;

	q0 = vk*cos(the_i); 
	q1 = vk*cos(the_s);                     
	q2 = vk*sqrt(eps_r - pow(sin(the_s), 2));
	q01 = vk*cos(the_i);                   
	q02 = vk*sqrt(eps_r - pow(sin(the_i), 2));

	C10 = (eps_r - 1.0) / (eps_r*q1 + q2) / (eps_r*q01 + q02);
	C20 = (-1.0)*(eps_r - 1.0) / (q1 + q2) / (q01 + q02);

	double ks[2] = { ksx, ksy };
	double k0[2] = { kix, kiy };
	dot_product_1(ks, k0, 2, ksk01);

	ksk0 = ksk01 / (vk*vk*sin(the_i)*sin(the_s));
	b1v = q2*q02*ksk0 - eps_r*vk*vk*sin(the_i)*sin(the_s);

	B1vv = C10*b1v;
	B1hh = C20*qv0*ksk0;
	double gt;
	double kiR = 0, ks0r = 0, kkr = 0;
	double g0 = 9.81;
	double k;

	int Num = 200;  
	int Num1 = 2 * Num + 1;

	double delt = 0.01;

	gt = Num1*delt / 6.0;

	complex<double> **WK;
	WK = (complex<double>**)malloc(sizeof(complex<double>*)* Num1);//
	for (int is = 0; is < Num1; is++)
	{
		WK[is] = (complex<double>*)malloc(sizeof(complex<double>)* Num1);
		for (int it = 0; it < Num1; it++)
		{
			WK[is][it] = complex<double>(0.0, 0.0);
		}
	}

	complex<double> **WK0;
	WK0 = (complex<double>**)malloc(sizeof(complex<double>*)* Num1);//
	for (int is = 0; is < Num1; is++)
	{
		WK0[is] = (complex<double>*)malloc(sizeof(complex<double>)* Num1);
		for (int it = 0; it < Num1; it++)
		{
			WK0[is][it] = complex<double>(0.0, 0.0);
		}
	}

	complex<double> **WK1;
	WK1 = (complex<double>**)malloc(sizeof(complex<double>*)* Num1);//
	for (int is = 0; is < Num1; is++)
	{
		WK1[is] = (complex<double>*)malloc(sizeof(complex<double>)* Num1);
		for (int it = 0; it < Num1; it++)
		{
			WK1[is][it] = complex<double>(0.0, 0.0);
		}
	}

	complex<double> **WK2;
	WK2 = (complex<double>**)malloc(sizeof(complex<double>*)* Num1);//
	for (int is = 0; is < Num1; is++)
	{
		WK2[is] = (complex<double>*)malloc(sizeof(complex<double>)* Num1);
		for (int it = 0; it < Num1; it++)
		{
			WK2[is][it] = complex<double>(0.0, 0.0);
		}
	}

	double k_x, k_y, k_k;

	double wpm = 0;
	double omg = 0;
	double temp1 = 0;
	complex<double> Sum1 = 0;
	complex<double> d_temp1 = 0, d_temp2 = 0, dd = 0;
	complex<double>BB_temp = 0, BB = 0;

	for (int i = 0; i < Num1; i++)
	{
		k_x = 2 * PI*(i - Num) / Num1 / delt;  
		for (int j = 0; j < Num1; j++)
		{
			k_y = 2 * PI*(j - Num) / Num1 / delt;
			Wpm(UU, k_x, k_y, wpm);  
			omega(k_x, k_y, omg);
			WK[i][j] = wpm*exp(complex<double>(0.0, omg*t));
			WK0[i][j] = wpm;
		}
	}
	Intk(Num, delt, WK0, Sum1); 

	//���ֺ�1
	complex<double> **Ker1;
	complex<double> **Ker2;
	complex<double> **Ker3;
	Ker1 = (complex<double>**)malloc(sizeof(complex<double>*)* Num1);//
	Ker2 = (complex<double>**)malloc(sizeof(complex<double>*)* Num1);
	Ker3 = (complex<double>**)malloc(sizeof(complex<double>*)* Num1);
	for (int is = 0; is < Num1; is++)
	{
		Ker1[is] = (complex<double>*)malloc(sizeof(complex<double>)* Num1);
		Ker2[is] = (complex<double>*)malloc(sizeof(complex<double>)* Num1);
		Ker3[is] = (complex<double>*)malloc(sizeof(complex<double>)* Num1);
		for (int it = 0; it < Num1; it++)
		{
			Ker1[is][it] = complex<double>(0.0, 0.0);
			Ker2[is][it] = complex<double>(0.0, 0.0);
			Ker3[is][it] = complex<double>(0.0, 0.0);
		}
	}
	complex<double> *row1;
	row1 = (complex<double>*)malloc(sizeof(complex<double>)* Num1);
	for (int it = 0; it < Num1; it++)
	{
		row1[it] = complex<double>(0.0, 0.0);
	}

	complex<double> *col1;
	col1 = (complex<double>*)malloc(sizeof(complex<double>)* Num1);
	for (int it = 0; it < Num1; it++)
	{
		col1[it] = complex<double>(0.0, 0.0);
	}

	complex<double> *row2;
	row2 = (complex<double>*)malloc(sizeof(complex<double>)* Num1);
	for (int it = 0; it < Num1; it++)
	{
		row2[it] = complex<double>(0.0, 0.0);
	}

	complex<double> *col2;
	col2 = (complex<double>*)malloc(sizeof(complex<double>)* Num1);
	for (int it = 0; it < Num1; it++)
	{
		col2[it] = complex<double>(0.0, 0.0);
	}

	int nrow, ncol;

	for (nrow = 0; nrow < Num1; nrow++)
	{
		for (ncol = 0; ncol < Num1; ncol++)
		{
			row1[ncol] = WK[nrow][ncol];
		}
		IFT_plus(Num, row1, row2);
		for (ncol = 0; ncol < Num1; ncol++)
		{
			Ker1[nrow][ncol] = row2[ncol];
		}
	}

	for (ncol = 0; ncol < Num1; ncol++)
	{
		for (nrow = 0; nrow < Num1; nrow++)
		{
			col1[nrow] = Ker1[nrow][ncol]; 
		}
		IFT_plus(Num, col1, col2);
		for (nrow = 0; nrow < Num1; nrow++)
		{
			Ker2[nrow][ncol] = pow(2 * PI / delt, 2)*col2[nrow]; 
		}
	}

	//���ֺ�2
	int ii, jj;
	for (ii = 0; ii < Num1; ii++)
	{
		for (jj = 0; jj < Num1; jj++)
		{
			double r[2] = { ii - Num, jj - Num };
			double rr = sqrt(r[0] * r[0] + r[1] * r[1]);
			double kr, kir, kir1;
			complex<double> G_xyz;
			dot_product_1(kspk0, r, 2, kr);

			G_xyz = exp(-(((ii - Num)*delt)*((ii - Num)*delt) + ((jj - Num)*delt)*((jj - Num)*delt)) / gt / gt / pow(cos(the_i), 2));

			Ker3[ii][jj] = (exp(pow(q0 + q01, 2)*real(Ker2[ii][jj])) - 1)*exp(-pow(q0 + q01, 2)*Sum1)*exp(complex<double>(0.0, -kr*delt))*G_xyz;
		}
	}


	complex<double> Sum2 = 0;
	Ints(Num, delt, Ker3, Sum2);

	b[0] = 4 * PI*PI / (pow(2 * PI, 4)) * 4 * q1*q01 / (pow(q1 + q01, 2))*pow(abs(B1vv), 2)*Sum2; 
	b[1] = 4 * PI*PI / (pow(2 * PI, 4)) * 4 * q1*q01 / (pow(q1 + q01, 2))*pow(abs(B1hh), 2)*Sum2;

}

int main()
{
	char Ansys[100];
	int nn = 1;
	complex<double> cj = complex<double>(0.0, 1.0);
	double t, j, m, the_i, the_s, phi_i, phi_s, vk, vw, lamda, freq, f0, i0;
	int i, ii;
	complex<double> b[2] = { 0 };
	double a[2] = { 0 };
	complex<double> dop[2] = { 0 };
	clock_t startTime, endTime;
	complex<double> eps_g = complex<double>(12, -16);
	startTime = clock();
	complex<double> sigmavv, sigmahh;
	double U = 5.0;
	double tt = 0;
	double delt_t = 0;
	
	freq = 1.228e9;

	lamda = VC / freq;
	vk = 2.0*PI / lamda;
	vw = 2.0 * PI*freq;
	double T, S;
	double sumvv = 0, sumhh = 0;

	complex<double> resultvv[nodetime + 1] = { 0, 0 };
	complex<double> resulthh[nodetime + 1] = { 0, 0 };
	complex<double> timevv[nodetime + 1] = { 0, 0 };
	complex<double> timehh[nodetime + 1] = { 0, 0 };
	double sfvv = 0;
	double sfhh = 0;
	double a1vv, a1hh;
	double a2vv, a2hh;
	double a3vv, a3hh;
	int f;
	phi_i = 0 * PI / 180.0;
	phi_s = 179.9999* PI / 180.0; 

	the_i = 30 * PI / 180.0;
	the_s = 30 * PI / 180.0;
	delt_t = 0.002;

	ofstream outfile_doppler;
	ofstream outfile_time;
        outfile_doppler.open("../output/doppler.txt", ios::trunc);
        outfile_time.open("../output//time.txt", ios::trunc);

        for (i = 0; i <= nodetime; i++) {
          tt = delt_t * i;
          SSA1_linear(eps_g, vk, the_s, phi_s, the_i, phi_i, U, tt, b);

          resultvv[i] = b[0];
          resulthh[i] = b[1];
          timevv[i] = resultvv[i] / resultvv[0];
          timehh[i] = resulthh[i] / resulthh[0];
          a1vv = real(timevv[i]);
          a2vv = imag(timevv[i]);
          a3vv = abs(timevv[i]);
          a1hh = real(timehh[i]);
          a2hh = imag(timehh[i]);
          a3hh = abs(timehh[i]);
          cout << tt << "  " << a1vv << "  " << a2vv << "  " << a3vv << endl;
          outfile_time << tt * 1000 << "        " << a1vv << "        " << a2vv
                       << "        " << a3vv << "        " << a1hh << "        "
                       << a2hh << "        " << a3hh << endl;
        }
        for (f = -200; f <= 200; f++) {
          complex<double> tempvv = {0.0, 0.0};
          complex<double> temphh = {0.0, 0.0};

          for (i = 1; i <= nodetime; i++) {
            tempvv = tempvv +
                     delt_t * timevv[i] *
                         exp(complex<double>(0.0, -2 * PI * f * delt_t * i));
            temphh = temphh +
                     delt_t * timehh[i] *
                         exp(complex<double>(0.0, -2 * PI * f * delt_t * i));
          }
          sfvv = delt_t * real(timevv[0]) + 2 * real(tempvv);
          sfhh = delt_t * real(timehh[0]) + 2 * real(temphh);
          outfile_doppler << f << "      " << sfvv << "      " << sfhh << endl;
          cout << f << "  " << sfvv << "  " << sfhh << endl;
        }
        outfile_doppler.close();
        outfile_time.close();
        /*************************************************/
        endTime = clock();
        t = (double)(endTime - startTime) / CLOCKS_PER_SEC / 60.0;
        return 0;
}
