#include "Fun.h"

#include <cmath>

#define PI 3.1415926535897932384626433832795

//��������S1����������S2/////////

double S1(double k, double u) {
  double a0 = 0.0014;
  double b = 0.74;
  double g = 981.;
  double s1;
  s1 = a0 * exp(-b * g * g / (k * k * pow(u, 4))) / pow(k, 3);
  return s1;
}
double S2(double k, double uf) {
  double p = 5 - log10(uf);
  double km = 3.63;
  double g = 981;
  double s2 = 0.875 * pow(2 * PI, p - 1) * (1 + 3 * k * k / (km * km)) *
              pow(g, (1 - p) / 2) *
              pow(k * (1 + k * k / (km * km)), -(p + 1) / 2);
  return s2;
}
double Wk(double k, double uf) {
  if (uf < 1.e-10) return 0.;
  if (k < 0.04)
    return S1(k, Uh(uf));
  else
    return S2(k, uf);
}
//������Ħ������ת��
double Uh(double uf) {
  if (uf < 1.e-10) return 0.;

  double u =
      (uf / 0.4) * log(1950. / (0.684 / uf + 4.28 * 1e-5 * uf * uf - 0.0443));
  return u;
}
//��ά����
double skf(double k, double siti, double uf) {
  double p =
      0.5 + 0.82 * exp(-0.5 * pow((sqrt(981. * k) / (856.5 / Uh(uf))), 4));
  double q = 0.32 * exp(-0.5 * pow((sqrt(981. * k) / (856.5 / Uh(uf))), 4));
  return Wk(k, uf) *
         (1 + p * cos(2 * siti) + q * cos(4 * siti));  // Wk��sigma������Ϊskp
}
