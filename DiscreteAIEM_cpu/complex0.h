/***********************complex_function*******************************/
#pragma once
#ifndef _COMplex_H_
#define _COMplex_H_

#include <math.h>

//#include "D:\fINITE_ELEMENT\alloc.h"
struct complex0 {
  double x, y;
};

typedef struct complex0 complex0;

complex0 COMplex_Cmplx(double x, double y);

complex0 COMplex_Conjg(complex0 z);

complex0 COMplex_Add(complex0 a, complex0 b);

complex0 COMplex_Add2(complex0 a, complex0 b, complex0 c);

complex0 COMplex_Sub(complex0 a, complex0 b);

//*******************************************************************
complex0 COMplex_Mul(complex0 a, complex0 b);

complex0 Real_Mul(double a, complex0 z);

complex0 COMplex_Div(complex0 a, complex0 b);

complex0 Real_Div(complex0 z, double a);

//*******************************************************************
double Real(complex0 z);

double Aimag(complex0 z);

//*******************************************************************
double COMplex_Abs(complex0 z);

complex0 COMplex_Expon(double a, complex0 z);

//*******************************************************************
complex0 COMplex_Sqrt(complex0 z);  // Added at Dec. 10

//*******************************************************************
complex0 COMplex_shuSub1(complex0 a, double b);

complex0 COMplex_shuSub2(complex0 a, double b);

complex0 COMplex_shuAdd(complex0 a, double b);

complex0 COMplex_Pow(complex0 a, double b);

//*******************************************************************

complex0 COMplex_Null();

void COMplex_Null_Vector(complex0 *V, int n);

void COMplex_Null_Matrix(complex0 **V, int nRow, int nCol);

void INT_Null_Vector(int *V, int n);

void INT_Null_Matrix(int **V, int nRow, int nCol);

/***************************/
#endif
