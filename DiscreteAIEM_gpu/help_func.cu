#include <cuComplex.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <chrono>
#include <complex>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

#include "main.h"

using std::complex;
using std::cout;
using std::endl;

#define PI 3.1415926535897932384626433832795
#define VC 2.99792458e8
#define nodetime 200
#define NUM 200

#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA error in %s:%d - %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                 \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

// Device functions
__device__ void d_Wpm(double U, double kx, double ky, double& skf) {
  double a = 8.1 * pow(10, -3);
  double b = 0.74;
  double g = 9.81;
  double Exp = exp(-b * g * g / (kx * kx + ky * ky + 1e-6) / pow(U, 4));
  double phik = pow(((kx / sqrt(kx * kx + ky * ky + 1e-10)) + 1) / 2, 2);
  // double phik = 1.0 / 4.0 * pow((1 + ky / sqrt(kx * kx + ky * ky + 1e-10)),
  // 2);//��� double phik = pow((1.0 - kx / sqrt(kx*kx + ky*ky + pow(10, -10)))
  // / 2.0, 2);//˳�� double phik = pow(((kx / sqrt(kx*kx + ky*ky + 1e-10)) + 1)
  // / 2, 2);//���
  skf = a * Exp * phik / pow((kx * kx + ky * ky + 1e-10), 2) / 2;
  // skf = phik / pow((kx * kx + ky * ky + 1e-10), 2) / 2;
}

__device__ void d_omega(double kx, double ky, double& omg) {
  double g = 9.81;
  double k = kx * kx + ky * ky;
  omg = sqrt(g * sqrt(k) * (1.0 + k / 363 / 363));
}

__device__ void d_dot_product_1(double* a, double* b, int size, double& n) {
  n = 0;
  for (int i = 0; i < size; i++) {
    n += a[i] * b[i];
  }
}

__device__ void d_K(double* KX, double* KY) {
  int index = 0;
  for (int i = 0; i < 2 * NUM + 1; i++) {
    for (int j = 0; j < 2 * NUM + 1; j++) {
      double Delt = 0.03;
      KX[index] = 2 * PI * (i - NUM) / (2 * NUM + 1) / Delt;
      KY[index] = 2 * PI * (j - NUM) / (2 * NUM + 1) / Delt;
      index++;
    }
  }
}

__device__ cuDoubleComplex d_exp_complex(double imag) {
  return make_cuDoubleComplex(cos(imag), sin(imag));
}

// CUDA kernels
__global__ void kernel_compute_WK(cuDoubleComplex* WK, cuDoubleComplex* WK0,
                                  double UU, double t, int Num, double delt,
                                  int Num1) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < Num1 && j < Num1) {
    double k_x = 2 * PI * (i - Num) / Num1 / delt;
    double k_y = 2 * PI * (j - Num) / Num1 / delt;
    double wpm, omg;
    d_Wpm(UU, k_x, k_y, wpm);
    d_omega(k_x, k_y, omg);

    int idx = i * Num1 + j;
    WK[idx] = cuCmul(make_cuDoubleComplex(wpm, 0.0), d_exp_complex(omg * t));
    WK0[idx] = make_cuDoubleComplex(wpm, 0.0);
    // printf("%g  %g\n", k_x, k_y);
    // printf("WK0= %g  %g\n", WK0[idx].x, WK0[idx].y);
  }
}

__global__ void kernel_2D_IFT_and_Ker(cuDoubleComplex* WK,
                                      cuDoubleComplex* Ker3, double* kspk0,
                                      double delt, double gt, double the_i,
                                      double vk, cuDoubleComplex Sum1, int Num,
                                      int Num1) {
  extern __shared__ cuDoubleComplex shared_data[];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row = bx * blockDim.x + tx;
  int col = by * blockDim.y + ty;

  if (row < Num1 && col < Num1) {
    // First IFT (row-wise)
    cuDoubleComplex sum1 = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < Num1; i++) {
      double phase = -2.0 * PI * i * row / Num1;
      sum1 = cuCadd(sum1, cuCmul(WK[i * Num1 + col], d_exp_complex(phase)));
      // printf("WK= %g %g\n", WK[i * Num1 + col].x, WK[i * Num1 + col].y);
    }

    double scale1 = 1.0 / sqrt((double)Num1);
    int out_row = (row < Num) ? row : (row - Num + Num);

    double phase_shift1 = 2.0 * PI * Num * (out_row + Num + 1) / Num1;
    cuDoubleComplex ker1 = cuCmul(make_cuDoubleComplex(scale1, 0.0),
                                  cuCmul(sum1, d_exp_complex(phase_shift1)));
    // printf("ker1= %g %g\n", ker1.x, ker1.y);
    // Store in shared memory for second IFT
    shared_data[ty * blockDim.x + tx] = ker1;
    __syncthreads();

    // Second IFT (column-wise)
    cuDoubleComplex sum2 = make_cuDoubleComplex(0.0, 0.0);
    for (int j = 0; j < Num1; j++) {
      double phase = -2.0 * PI * j * col / Num1;
      cuDoubleComplex val =
          (j / blockDim.y == by)
              ? shared_data[(j % blockDim.y) * blockDim.x + tx]
              : make_cuDoubleComplex(0.0, 0.0);
      sum2 = cuCadd(sum2, cuCmul(val, d_exp_complex(phase)));
    }

    double scale2 = pow(2 * PI / delt, 2) / sqrt((double)Num1);
    int out_col = (col < Num) ? col : (col - Num + Num);
    double phase_shift2 = 2.0 * PI * Num * (out_col + Num + 1) / Num1;
    cuDoubleComplex ker2 = cuCmul(make_cuDoubleComplex(scale2, 0.0),
                                  cuCmul(sum2, d_exp_complex(phase_shift2)));
    // printf("ker2= %g %g\n", ker2.x, ker2.y);
    // Compute Ker3 directly
    double r[2] = {(double)(out_row - Num), (double)(out_col - Num)};
    double kr;
    double q0 = vk * cos(the_i);
    // printf("q0= %g\n", pow(q0 + q0, 2));
    d_dot_product_1(kspk0, r, 2, kr);

    double G_xyz = exp(-((out_row - Num) * delt * (out_row - Num) * delt +
                         (out_col - Num) * delt * (out_col - Num) * delt) /
                       (gt * gt * pow(cos(the_i), 2)));
    // double T1 = exp(pow(q0 + q0, 2) * cuCreal(ker2));
    // printf("q0= %g\n", T1);
    // cuDoubleComplex ker2_real = make_cuDoubleComplex(T1, 0.0);
    cuDoubleComplex ker2_real = make_cuDoubleComplex(cuCreal(ker2), 0.0);
    // cuDoubleComplex ker2_real = cuCsub(d_exp_complex(pow(q0 + q0, 2)*
    // cuCreal(ker2)), make_cuDoubleComplex(1.0,0.0)); cuDoubleComplex ker2_real
    // = cuCsub(T1_term, make_cuDoubleComplex(1.0, 0.0));
    cuDoubleComplex exp_term = d_exp_complex(-kr * delt);
    cuDoubleComplex g_xyz_term = make_cuDoubleComplex(G_xyz, 0.0);

    cuDoubleComplex temp1 = cuCmul(ker2_real, Sum1);
    // cuDoubleComplex temp1 = cuCmul(ker2_real, d_exp_complex(-pow(q0 + q0, 2)*
    // cuCreal(Sum1)));
    cuDoubleComplex temp2 = cuCmul(exp_term, g_xyz_term);
    Ker3[out_row * Num1 + out_col] = cuCmul(temp1, temp2);
  }
}

__global__ void kernel_Int(cuDoubleComplex* Matrix_int,
                           cuDoubleComplex* partial_sums, double factor,
                           int Num1) {
  extern __shared__ cuDoubleComplex shared_data[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  cuDoubleComplex local_sum = make_cuDoubleComplex(0.0, 0.0);

  while (idx < Num1 * Num1) {
    local_sum = cuCadd(
        local_sum, cuCmul(Matrix_int[idx], make_cuDoubleComplex(factor, 0.0)));
    idx += blockDim.x * gridDim.x;
  }

  shared_data[tid] = local_sum;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_data[tid] = cuCadd(shared_data[tid], shared_data[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_sums[blockIdx.x] = shared_data[0];
  }
}

// Main function
void SSA1_linear(complex<double> eps_r, double vk, double the_s, double phi_s,
                 double the_i, double phi_i, double UU, double t,
                 complex<double>* b) {
  double q0, q01, q1, vw, vc = 2.99792458e8, vr2, qv0, qv1, Pinc, ksk01 = 0,
                          ksk0, qx, qy;
  complex<double> q02, q2, C10, C20, B1vv, B1hh, P_inc, p1p2;
  complex<double> b1v, eps_g;
  double k0x, k0y, ksx, ksy, kix, kiy, kiz, ksk00;
  double freq = vk * VC / 2 / PI;

  k0x = vk * sin(the_i) * cos(phi_i);
  k0y = vk * sin(the_i) * sin(phi_i);
  ksx = vk * sin(the_s) * cos(phi_s);
  ksy = vk * sin(the_s) * sin(phi_s);
  kix = vk * sin(the_i) * cos(phi_i);
  kiy = vk * sin(the_i) * sin(phi_i);
  qx = ksx - k0x;
  qy = ksy - k0y;
  double kspk0[2] = {qx, qy};
  vw = 2 * PI * freq;  //���䲨��Ƶ��
  qv0 = vw * vw / vc / vc;

  q0 = vk * cos(the_i);                        // wc
  q1 = vk * cos(the_s);                        //ɢ�䲨��ֱ����
  q2 = vk * sqrt(eps_r - pow(sin(the_s), 2));  //ɢ�䲨�ڽ����еĴ�ֱ����
  q01 = vk * cos(the_i);  //���䲨��ֱ����
  q02 = vk * sqrt(eps_r - pow(sin(the_i), 2));  //���䲨�ڽ����еĴ�ֱ����

  C10 = (eps_r - 1.0) / (eps_r * q1 + q2) / (eps_r * q01 + q02);  // Bvv_ǰ����
  C20 = (-1.0) * (eps_r - 1.0) / (q1 + q2) / (q01 + q02);  // Bhh_ǰ����

  double ks[2] = {ksx, ksy};
  double k0[2] = {kix, kiy};
  dot_product_1(ks, k0, 2, ksk01);

  ksk0 = ksk01 / (vk * vk * sin(the_i) * sin(the_s));  //��ks��k0��/|ks��k0|
  b1v =
      q2 * q02 * ksk0 - eps_r * vk * vk * sin(the_i) * sin(the_s);  // Bvv_�����

  B1vv = C10 * b1v;
  B1hh = C20 * qv0 * ksk0;
  // std::cout << "B1hh=" << B1hh << endl;
  // system("pause");
  double lamda = vc / freq;

  double kiR = 0, ks0r = 0, kkr = 0;
  double g0 = 9.81;
  double k;

  double LLL = 200 * PI * lamda;
  double sigma = 0;

  int Num = 200;
  int Num1 = 2 * Num + 1;
  double delt = 0.01;
  double gt = Num1 * delt / 6.0;
  // Device memory allocation
  cuDoubleComplex *d_WK, *d_WK0, *d_Ker3, *d_Sum;
  double* d_kspk0;
  double *d_KX, *d_KY;
  size_t size = Num1 * Num1 * sizeof(cuDoubleComplex);

  CUDA_CHECK(cudaMalloc(&d_WK, size));
  CUDA_CHECK(cudaMalloc(&d_WK0, size));
  CUDA_CHECK(cudaMalloc(&d_Ker3, size));
  CUDA_CHECK(cudaMalloc(&d_Sum, 256 * sizeof(cuDoubleComplex)));
  CUDA_CHECK(cudaMalloc(&d_kspk0, 2 * sizeof(double)));

  CUDA_CHECK(
      cudaMemcpy(d_kspk0, kspk0, 2 * sizeof(double), cudaMemcpyHostToDevice));

  // Use 16x16 blocks for optimal GPU utilization
  dim3 blockDim(16, 16);
  dim3 gridDim((Num1 + blockDim.x - 1) / blockDim.x,
               (Num1 + blockDim.y - 1) / blockDim.y);

  // Compute WK and WK0 in one kernel
  kernel_compute_WK<<<gridDim, blockDim>>>(d_WK, d_WK0, UU, t, Num, delt, Num1);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Compute Sum1
  int threads = 256;
  int blocks = min(256, (Num1 * Num1 + threads - 1) / threads);
  kernel_Int<<<blocks, threads, threads * sizeof(cuDoubleComplex)>>>(
      d_WK0, d_Sum, pow(2 * PI / Num1 / delt, 2), Num1);
  CUDA_CHECK(cudaDeviceSynchronize());

  cuDoubleComplex h_Sum[256];
  CUDA_CHECK(cudaMemcpy(h_Sum, d_Sum, blocks * sizeof(cuDoubleComplex),
                        cudaMemcpyDeviceToHost));
  cuDoubleComplex Sum1 = make_cuDoubleComplex(0.0, 0.0);
  for (int i = 0; i < blocks; i++) {
    Sum1 = cuCadd(Sum1, h_Sum[i]);
  }
  // std::cout <<"ʵ��=" << Sum1.x << "�鲿=" << Sum1.y << endl;
  // system("pause");

  // Compute 2D IFT and Ker3 in one kernel
  size_t shared_mem_size = blockDim.x * blockDim.y * sizeof(cuDoubleComplex);
  kernel_2D_IFT_and_Ker<<<gridDim, blockDim, shared_mem_size>>>(
      d_WK, d_Ker3, d_kspk0, delt, gt, the_i, vk, Sum1, Num, Num1);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Compute Sum2
  kernel_Int<<<blocks, threads, threads * sizeof(cuDoubleComplex)>>>(
      d_Ker3, d_Sum, pow(delt, 2), Num1);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_Sum, d_Sum, blocks * sizeof(cuDoubleComplex),
                        cudaMemcpyDeviceToHost));
  cuDoubleComplex Sum2 = make_cuDoubleComplex(0.0, 0.0);
  for (int i = 0; i < blocks; i++) {
    Sum2 = cuCadd(Sum2, h_Sum[i]);
  }
  // std::cout << "ʵ��=" << Sum2.x << "�鲿=" << Sum2.y << endl;
  // system("pause");
  // Final computation
  // complex<double> factor = 4 * PI * PI / pow(2 * PI, 4) * 4 * q1 * q01 /
  // (pow(q1 + q01, 2)) * pow(abs(B1vv), 2);
  complex<double> factor = 4 * PI * PI / pow(2 * PI, 4);
  complex<double> Sum2_host(cuCreal(Sum2), cuCimag(Sum2));
  // cout << Sum2_host << endl;
  // system("pause");
  b[0] = factor * Sum2_host;
  b[1] = factor * Sum2_host;

  // Cleanup
  CUDA_CHECK(cudaFree(d_WK));
  CUDA_CHECK(cudaFree(d_WK0));
  CUDA_CHECK(cudaFree(d_Ker3));
  CUDA_CHECK(cudaFree(d_Sum));
  CUDA_CHECK(cudaFree(d_kspk0));
}

// Host helper functions
void Wpm(double U, double kx, double ky, double& skf) {
  double phik = pow(((kx / sqrt(kx * kx + ky * ky + 1e-10)) + 1) / 2, 2);
  skf = phik / pow((kx * kx + ky * ky + 1e-10), 2) / 2;
}

void omega(double kx, double ky, double& omg) {
  double g = 9.81;
  double k = kx * kx + ky * ky;
  omg = sqrt(g * sqrt(k) * (1.0 + k / 363 / 363));
}

void dot_product_1(double* a, double* b, int size, double& n) {
  n = 0;
  for (int i = 0; i < size; i++) {
    n += a[i] * b[i];
  }
}
