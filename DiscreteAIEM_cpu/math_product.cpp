
#include <math.h>

#include <complex>
#include <iostream>

using namespace std;

void add_product(double *a, double *b, double c[2]) {
  c[0] = a[0] + b[0];
  c[1] = a[1] + b[1];
}
void subtract_product(double *a, double *b, double c[2]) {
  c[0] = a[0] - b[0];
  c[1] = a[1] - b[1];
}

void dot_product_1(double *a, double *b, int size, double &n) {
  int i;
  n = 0;
  for (i = 0; i < size; i++) {
    n += a[i] * b[i];
  }
}

void cross_product_1(double *a, double *b, double c[3]) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

void dot_productc_1(complex<double> *a, complex<double> *b, int size,
                    complex<double> &n) {
  int i;
  n = complex<double>(0, 0);
  for (i = 0; i < size; i++) {
    n = n + a[i] * b[i];
  }
}

void cross_productc_1(complex<double> *a, complex<double> *b,
                      complex<double> c[3]) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

void dot_productx_1(double *a, complex<double> *b, int size,
                    complex<double> &n) {
  int i;
  n = complex<double>(0, 0);
  for (i = 0; i < size; i++) {
    n = n + a[i] * b[i];
  }
}

void cross_productx1_1(double *a, complex<double> *b, complex<double> c[3]) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

void cross_productx2_1(complex<double> *b, double *a, complex<double> c[3]) {
  c[0] = a[2] * b[1] - a[1] * b[2];
  c[1] = a[0] * b[2] - a[2] * b[0];
  c[2] = a[1] * b[0] - a[0] * b[1];
}
