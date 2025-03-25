#ifndef __MINE_H
#define __MINE_H
#include <math.h>
#include <iostream>
#include <complex>

using namespace std;

double dot_product();
double *cross_product();
complex<double> dot_productc();
complex<double> *cross_productc();
complex<double> dot_productx();
complex<double> *cross_productx1();
complex<double> *cross_productx2();
void add_product(double *a, double *b, double c[2]);
void subtract(double *a, double *b,double c[2]);
void dot_product_1(double *a, double *b, int size, double &n);
void cross_product_1(double *a, double *b, double c[3]);
void dot_productc_1(double *a, double *b, int size, double &n);
void cross_productc_1(complex<double> *a, complex<double> *b, complex<double> c[3]);
void dot_productx_1(double *a, complex<double> *b, int size, complex<double> &n);
void cross_productx1_1(double *a, complex<double> *b, complex<double> c[3]);
void cross_productx2_1(complex<double> *b, double *a, complex<double> c[3]);


double dot_product(double *a, double *b, int size)
{
	int i;
	double n = 0;
	for (i = 0; i < size; i++)
	{
		n += a[i] * b[i];
	}
	return n;
}

double *cross_product(double *a, double *b)
{
	int size = 3;
	double *c = (double*)malloc(sizeof(double)*size);
	memset(c, 0, size*sizeof(double));
	c[0] = a[1] * b[2] - a[2] * b[1];
	c[1] = a[2] * b[0] - a[0] * b[2];
	c[2] = a[0] * b[1] - a[1] * b[0];
	return c;
}

complex<double> dot_productc(complex<double> *a, complex<double> *b, int size)
{
	int i;
	complex<double> n = complex<double>(0, 0);
	for (i = 0; i < size; i++)
	{
		n = n + a[i] * b[i];
	}
	return n;
}

complex<double> *cross_productc(complex<double> *a, complex<double> *b)
{
	int size = 3;
	complex<double> *c = (complex<double>*)malloc(sizeof(complex<double>)*size);
	*c = a[1] * b[2] - a[2] * b[1];
	*(c + 1) = a[2] * b[0] - a[0] * b[2];
	*(c + 2) = a[0] * b[1] - a[1] * b[0];
	return c;
}

complex<double> dot_productx(double *a, complex<double> *b, int size)
{
	int i;
	complex<double> n = complex<double>(0, 0);
	for (i = 0; i < size; i++)
	{
		n = n + a[i] * b[i];
	}
	return n;
}

complex<double> *cross_productx1(double *a, complex<double> *b)
{
	int size = 3;
	complex<double> *c = (complex<double>*)malloc(sizeof(complex<double>)*size);
	*c = a[1] * b[2] - a[2] * b[1];
	*(c + 1) = a[2] * b[0] - a[0] * b[2];
	*(c + 2) = a[0] * b[1] - a[1] * b[0];
	return c;
}

complex<double> *cross_productx2(complex<double> *b, double *a)
{
	int size = 3;
	complex<double> *c = (complex<double>*)malloc(sizeof(complex<double>)*size);
	*c = a[2] * b[1] - a[1] * b[2];
	*(c + 1) = a[0] * b[2] - a[2] * b[0];
	*(c + 2) = a[1] * b[0] - a[0] * b[1];
	return c;
}



#endif