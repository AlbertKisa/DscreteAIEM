#include <fstream>
#include <iostream>
#include <sstream>

#include "cuda_runtime.h"
#include "main.h"

int main(int argc, char* argv[]) {
  char Ansys[100];
  int nn = 1;
  complex<double> cj = complex<double>(0.0, 1.0);
  double t, j, m, the_i, the_s, phi_i, phi_s, vk, vw, lamda, freq, f0, i0;
  int i, ii;
  complex<double> b[2] = {0};
  double a[2] = {0};
  complex<double> dop[2] = {0};
  clock_t startTime, endTime;
  complex<double> eps_g = complex<double>(12, -16);
  startTime = clock();
  complex<double> sigmavv, sigmahh;
  double U = 5.0;
  double tt = 0;
  double delt_t = 0;

  freq = 1.228e9;

  lamda = VC / freq;
  vk = 2.0 * PI / lamda;
  vw = 2.0 * PI * freq;
  double T, S;
  double sumvv = 0, sumhh = 0;

  complex<double> resultvv[nodetime + 1] = {0, 0};
  complex<double> resulthh[nodetime + 1] = {0, 0};
  complex<double> timevv[nodetime + 1] = {0, 0};
  complex<double> timehh[nodetime + 1] = {0, 0};
  std::ofstream outfile_doppler;
  std::ofstream outfile_time;
  outfile_doppler.open("../output/doppler.txt");
  outfile_time.open("../output//time.txt");
  double sfvv = 0;
  double sfhh = 0;
  double a1vv, a1hh;
  double a2vv, a2hh;
  double a3vv, a3hh;
  int f;
  phi_i = 0 * PI / 180.0;
  phi_s = 179.9999 * PI / 180.0;

  the_i = 30 * PI / 180.0;
  the_s = 30 * PI / 180.0;
  delt_t = 0.002;

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
                 << "        " << a3vv << endl;
  }
  for (f = -200; f <= 200; f++) {
    complex<double> tempvv = {0.0, 0.0};
    complex<double> temphh = {0.0, 0.0};
    for (i = 1; i <= nodetime; i++) {
      tempvv = tempvv + delt_t * timevv[i] *
                            exp(complex<double>(0.0, -2 * PI * f * delt_t * i));
      temphh = temphh + delt_t * timehh[i] *
                            exp(complex<double>(0.0, -2 * PI * f * delt_t * i));
    }
    sfvv = delt_t * real(timevv[0]) + 2 * real(tempvv);
    sfhh = delt_t * real(timehh[0]) + 2 * real(temphh);
    cout << f << "  " << sfvv << endl;
    outfile_doppler << f << "      " << sfvv << endl;
  }

  outfile_doppler.close();
  outfile_time.close();

  endTime = clock();
  t = (double)(endTime - startTime) / CLOCKS_PER_SEC / 60.0;
  printf("执行时间：%f min\n", t);

  return 0;
}
