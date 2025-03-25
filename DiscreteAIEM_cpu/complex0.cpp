#include "complex0.h"

int ii1, jj1;
complex0 COMplex_Cmplx(double x, double y)
// double x,y;
{
  complex0 z;
  z.x = x;
  z.y = y;
  return z;
}
complex0 COMplex_Conjg(complex0 z)
// complex z;
{
  return COMplex_Cmplx(z.x, -z.y);
}
complex0 COMplex_Add(complex0 a, complex0 b)
// complex a,b;
{
  return COMplex_Cmplx(a.x + b.x, a.y + b.y);
}
complex0 COMplex_Add2(complex0 a, complex0 b, complex0 c)
// complex a,b,c;
{
  return COMplex_Cmplx(a.x + b.x + c.x, a.y + b.y + c.y);
}
complex0 COMplex_Sub(complex0 a, complex0 b)
// complex a,b;
{
  return COMplex_Cmplx(a.x - b.x, a.y - b.y);
}
//*******************************************************************
complex0 COMplex_Mul(complex0 a, complex0 b)
// complex a,b;
{
  return COMplex_Cmplx(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
complex0 Real_Mul(double a, complex0 z)
// double a;complex z;
{
  return COMplex_Cmplx(a * z.x, a * z.y);
}
complex0 COMplex_Div(complex0 a, complex0 b)
// complex a,b;
{
  double D = b.x * b.x + b.y * b.y;
  return COMplex_Cmplx((a.x * b.x + a.y * b.y) / D,
                       (a.y * b.x - a.x * b.y) / D);
}
complex0 Real_Div(complex0 z, double a)
// double a;complex z;
{
  return COMplex_Cmplx(z.x / a, z.y / a);
}
//*******************************************************************
double Real(complex0 z)
// complex z;
{
  return z.x;
}
double Aimag(complex0 z)
// complex z;
{
  return z.y;
}
//*******************************************************************
double COMplex_Abs(complex0 z)
// complex z;
{
  return sqrt(z.x * z.x + z.y * z.y);
}
complex0 COMplex_Expon(double a, complex0 z)
// double a; complex z;
{
  double R = exp((a * z.x));
  return COMplex_Cmplx(R * cos(z.y), (a * R) * sin(z.y));
}
//*******************************************************************
complex0 COMplex_Sqrt(complex0 z)  // Added at Dec. 10
// complex z;
{
  complex0 c;
  float xx, yy, w, r;
  if ((z.x == 0.0) && (z.y == 0.0)) {
    c.x = 0.0;
    c.y = 0.0;
    return c;
  } else {
    xx = fabs(z.x);
    yy = fabs(z.y);
    if (xx >= yy) {
      r = yy / xx;
      w = sqrt(xx) * sqrt(0.5 * (1.0 + sqrt(1.0 + r * r)));
    } else {
      r = xx / yy;
      w = sqrt(yy) * sqrt(0.5 * (r + sqrt(1.0 + r * r)));
    }
    if (z.x >= 0.0) {
      c.x = w;
      c.y = z.y / (2.0 * w);
    } else {
      c.y = (z.y >= 0) ? w : -w;
      c.x = z.y / (2.0 * c.y);
    }
    return c;
  }
}
//*******************************************************************
complex0 COMplex_shuSub1(complex0 a, double b) {
  complex0 z;
  z.x = a.x - b;
  z.y = a.y;
  return z;
}
complex0 COMplex_shuSub2(complex0 a, double b) {
  complex0 z;
  z.x = b - a.x;
  z.y = -a.y;
  return z;
}
complex0 COMplex_shuAdd(complex0 a, double b) {
  complex0 z;
  z.x = a.x + b;
  z.y = a.y;
  return z;
}

complex0 COMplex_Pow(complex0 a, double b) {
  long i;
  complex0 z;
  if (b == 0.0) return COMplex_Cmplx(1.0, 0.0);
  if (b == 1.0) return a;
  if (b > 1.0) {
    z = a;
    for (i = 2.0; i <= b; i++) {
      z = COMplex_Mul(z, a);
    }
    return z;
  }
  if (b == -1.0) return COMplex_Div(COMplex_Cmplx(1.0, 0.0), a);
  if (b < -1.0) {
    z = COMplex_Div(COMplex_Cmplx(1.0, 0.0), a);
    for (i = 2.0; i <= abs(b); i++) {
      z = COMplex_Mul(z, COMplex_Div(COMplex_Cmplx(1.0, 0.0), a));
    }
    return z;
  }
}
//*******************************************************************

complex0 COMplex_Null() {
  complex0 z;
  z.x = 0.0;
  z.y = 0.0;
  return z;
}

void COMplex_Null_Vector(complex0 *V, int n) {  // V=CMPLX_Vector(n);
  for (ii1 = 0; ii1 < n; ii1++) V[ii1] = COMplex_Null();

  // free_CMPLX_Vector(V,n);
}

void COMplex_Null_Matrix(complex0 **V, int nRow,
                         int nCol) {  // V=CMPLX_Matrix(nRow,nCol);
  for (ii1 = 0; ii1 < nRow; ii1++)
    for (jj1 = 0; jj1 < nCol; jj1++) V[ii1][jj1] = COMplex_Null();

  // free_CMPLX_Matrix(V,nRow,nCol);
}

void INT_Null_Vector(int *V, int n) {  // V=INT_Vector(n);
  for (ii1 = 0; ii1 < n; ii1++) V[ii1] = 0;

  // free_INT_Vector(V,n);
}

void INT_Null_Matrix(int **V, int nRow, int nCol) {  // V=INT_Matrix(nRow,nCol);
  for (ii1 = 0; ii1 < nRow; ii1++)
    for (jj1 = 0; jj1 < nCol; jj1++) V[ii1][jj1] = 0;

  // free_INT_Matrix(V,nRow,nCol);
}
/***************************/
