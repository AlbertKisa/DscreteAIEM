// #include "main.h"




// void SSA1_linear(complex<double> eps_r, double vk, double the_s, double phi_s,
//     double the_i, double phi_i, double UU, double t, complex<double>* b) {
//     double qx, qy, qz;
//     double k0x, k0y, ksx, ksy;

//     k0x = vk * sin(the_i) * cos(phi_i);
//     k0y = vk * sin(the_i) * sin(phi_i);
//     ksx = vk * sin(the_s) * cos(phi_s);
//     ksy = vk * sin(the_s) * sin(phi_s);
//     qx = ksx - k0x;
//     qy = ksy - k0y;
//     double kspk0[2] = { qx, qy };

//     double g0 = 9.81;
//     double k;
//     double sigma = 0;

//     int Num = 200;
//     int Num1 = 2 * Num + 1;

//     double delt = 0.03;
//     double gt = Num1 * delt / 6.0;

//     complex<double>** WK;
//     WK = (complex<double>**)malloc(sizeof(complex<double>*) * Num1);
//     for (int is = 0; is < Num1; is++) {
//         WK[is] = (complex<double>*)malloc(sizeof(complex<double>) * Num1);
//         for (int it = 0; it < Num1; it++) {
//             WK[is][it] = complex<double>(0.0, 0.0);
//         }
//     }

//     complex<double>** WK0;
//     WK0 = (complex<double>**)malloc(sizeof(complex<double>*) * Num1);
//     for (int is = 0; is < Num1; is++) {
//         WK0[is] = (complex<double>*)malloc(sizeof(complex<double>) * Num1);
//         for (int it = 0; it < Num1; it++) {
//             WK0[is][it] = complex<double>(0.0, 0.0);
//         }
//     }

//     complex<double>** WK1;
//     WK1 = (complex<double>**)malloc(sizeof(complex<double>*) * Num1);
//     for (int is = 0; is < Num1; is++) {
//         WK1[is] = (complex<double>*)malloc(sizeof(complex<double>) * Num1);
//         for (int it = 0; it < Num1; it++) {
//             WK1[is][it] = complex<double>(0.0, 0.0);
//         }
//     }

//     complex<double>** WK2;
//     WK2 = (complex<double>**)malloc(sizeof(complex<double>*) * Num1);
//     for (int is = 0; is < Num1; is++) {
//         WK2[is] = (complex<double>*)malloc(sizeof(complex<double>) * Num1);
//         for (int it = 0; it < Num1; it++) {
//             WK2[is][it] = complex<double>(0.0, 0.0);
//         }
//     }

//     double k_x, k_y, k_k;

//     double wpm = 0;
//     double omg = 0;
//     double temp1 = 0;
//     complex<double> Sum1 = 0;
//     complex<double> d_temp1 = 0, d_temp2 = 0, dd = 0;
//     complex<double>BB_temp = 0, BB = 0;

//     for (int i = 0; i < Num1; i++) {
//         k_x = 2 * PI * (i - Num) / Num1 / delt;
//         for (int j = 0; j < Num1; j++) {
//             k_y = 2 * PI * (j - Num) / Num1 / delt;
//             Wpm(UU, k_x, k_y, wpm);
//             omega(k_x, k_y, omg);
//             WK[i][j] = wpm * exp(complex<double>(0.0, omg * t));
//             WK0[i][j] = wpm;
//         }
//     }
//     Intk(Num, delt, WK0, Sum1);

//     complex<double>** Ker1;
//     complex<double>** Ker2;
//     complex<double>** Ker3;
//     Ker1 = (complex<double>**)malloc(sizeof(complex<double>*) * Num1);//
//     Ker2 = (complex<double>**)malloc(sizeof(complex<double>*) * Num1);
//     Ker3 = (complex<double>**)malloc(sizeof(complex<double>*) * Num1);
//     for (int is = 0; is < Num1; is++)
//     {
//         Ker1[is] = (complex<double>*)malloc(sizeof(complex<double>) * Num1);
//         Ker2[is] = (complex<double>*)malloc(sizeof(complex<double>) * Num1);
//         Ker3[is] = (complex<double>*)malloc(sizeof(complex<double>) * Num1);
//         for (int it = 0; it < Num1; it++)
//         {
//             Ker1[is][it] = complex<double>(0.0, 0.0);
//             Ker2[is][it] = complex<double>(0.0, 0.0);
//             Ker3[is][it] = complex<double>(0.0, 0.0);
//         }
//     }
//     complex<double>* row1;
//     row1 = (complex<double>*)malloc(sizeof(complex<double>) * Num1);
//     for (int it = 0; it < Num1; it++)
//     {
//         row1[it] = complex<double>(0.0, 0.0);
//     }

//     complex<double>* col1;
//     col1 = (complex<double>*)malloc(sizeof(complex<double>) * Num1);
//     for (int it = 0; it < Num1; it++)
//     {
//         col1[it] = complex<double>(0.0, 0.0);
//     }

//     complex<double>* row2;
//     row2 = (complex<double>*)malloc(sizeof(complex<double>) * Num1);
//     for (int it = 0; it < Num1; it++)
//     {
//         row2[it] = complex<double>(0.0, 0.0);
//     }

//     complex<double>* col2;
//     col2 = (complex<double>*)malloc(sizeof(complex<double>) * Num1);
//     for (int it = 0; it < Num1; it++)
//     {
//         col2[it] = complex<double>(0.0, 0.0);
//     }

//     int nrow, ncol;

//     for (nrow = 0; nrow < Num1; nrow++)
//     {
//         for (ncol = 0; ncol < Num1; ncol++)
//         {
//             row1[ncol] = WK[nrow][ncol];
//         }
//         IFT_plus(Num, row1, row2);
//         for (ncol = 0; ncol < Num1; ncol++)
//         {
//             Ker1[nrow][ncol] = row2[ncol];
//         }
//     }

//     for (ncol = 0; ncol < Num1; ncol++)
//     {
//         for (nrow = 0; nrow < Num1; nrow++)
//         {
//             col1[nrow] = Ker1[nrow][ncol];
//         }
//         IFT_plus(Num, col1, col2);
//         for (nrow = 0; nrow < Num1; nrow++)
//         {
//             Ker2[nrow][ncol] = pow(2 * PI / delt, 2) * col2[nrow];
//         }
//     }

//     int ii, jj;
//     for (ii = 0; ii < Num1; ii++)
//     {
//         for (jj = 0; jj < Num1; jj++)
//         {
//             double r[2] = { ii - Num, jj - Num };
//             double rr = sqrt(r[0] * r[0] + r[1] * r[1]);
//             double kr, kir, kir1;
//             complex<double> G_xyz;
//             dot_product_1(kspk0, r, 2, kr);
//             G_xyz = exp(-(((ii - Num) * delt) * ((ii - Num) * delt) + ((jj - Num) * delt) * ((jj - Num) * delt)) / gt / gt / pow(cos(the_i), 2));
//             Ker3[ii][jj] = real(Ker2[ii][jj]) * Sum1 * exp(complex<double>(0.0, -kr * delt)) * G_xyz;
//         }
//     }
//     complex<double> Sum2 = 0;
//     Ints(Num, delt, Ker3, Sum2);
//     b[0] = 4 * PI * PI / (pow(2 * PI, 4)) * Sum2;
//     b[1] = 4 * PI * PI / (pow(2 * PI, 4)) * Sum2;
// }

// void Wpm(double U, double kx, double ky, double& skf) {
//     double phik;
//     phik = pow(((kx / sqrt(kx * kx + ky * ky + pow(10, -10))) + 1) / 2, 2);
//     skf = phik / pow((kx * kx + ky * ky + pow(10, -10)), 2) / 2;
// }

// void omega(double kx, double ky, double& omg) {
//     double g, k;
//     g = 9.81;
//     k = kx * kx + ky * ky;
//     omg = sqrt(g * sqrt(k) * (1.0 + k / 363 / 363));
// }
// void Intk(int Num, double delt, complex<double>** Matrix_int, complex<double>& Sum) {
//     Sum = 0;
//     for (int i = 0; i < 2 * Num + 1; i++) {
//         for (int j = 0; j < 2 * Num + 1; j++) {
//             Sum = Sum + Matrix_int[i][j] * pow(2 * PI / (2 * Num + 1) / delt, 2);
//         }
//     }
// }
// void Ints(int Num, double delt, complex<double>** Matrix_int, complex<double>& Sum) {
//     Sum = 0;
//     for (int i = 0; i < 2 * Num + 1; i++) {
//         for (int j = 0; j < 2 * Num + 1; j++) {
//             Sum = Sum + Matrix_int[i][j] * pow(delt, 2);
//         }
//     }
// }
// void IFT_mathematica(int Num, complex<double>* Matrix_ori, complex<double>* ISAR1_t1) {
//     for (int k = 0; k < Num; k++) {
//         complex<double> sum_1dft = complex<double>(0.0, 0.0);
//         for (int i = 0; i < Num; i++) {
//             complex<double> Phase = complex<double>(0.0, -2 * (i)*PI / Num * k);
//             sum_1dft = sum_1dft + Matrix_ori[i] * exp(Phase);
//         }
//         if (1) {
//             ISAR1_t1[k] = (1.0 / sqrt(Num)) * sum_1dft;

//         }
//     }
// }
// void IFT_plus(int Num, complex<double>* Matrix_int, complex<double>* Matrix_out) {
//     int Num1 = 2 * Num + 1;
//     complex<double>* Matrix_temp;
//     Matrix_temp = (complex<double>*)malloc(sizeof(complex<double>) * Num1);//
//     IFT_mathematica(Num1, Matrix_int, Matrix_temp);
//     for (int i = 0; i < Num; i++) {
//         Matrix_out[i] = 1.0 / sqrt(Num1) * exp(complex<double>(0, 2 * PI * Num / Num1 * (i + Num + 1))) * Matrix_temp[i + Num + 1];
//     }
//     for (int j = Num; j < Num1; j++) {
//         Matrix_out[j] = 1.0 / sqrt(Num1) * exp(complex<double>(0, 2 * PI * Num / Num1 * (j + Num + 1))) * Matrix_temp[j - Num];
//     }
//     free(Matrix_temp);
//     Matrix_temp = NULL;
// }
// void dot_product_1(double* a, double* b, int size, double& n) {
//     int i;
//     n = 0;
//     for (i = 0; i < size; i++) {
//         n += a[i] * b[i];
//     }
// }







// #include "main.h"
// #include <iostream>
// #include <stdio.h>
// #include <string.h>
// #include <stdlib.h>
// #include <math.h>
// #include <time.h>
// #include <complex>
// #include <ctime>
// #include <random>
// #include <chrono>
// #include <fstream>
// #include <sstream>
// #include <cuda_runtime.h>
// #include <cuComplex.h>
// #include <device_launch_parameters.h>

// using std::complex;
// using std::cout;
// using std::endl;

// #define PI          3.1415926535897932384626433832795
// #define VC          2.99792458e8 
// #define nodetime    200

// #define CUDA_CHECK(call) \
//     do { \
//         cudaError_t err = call; \
//         if (err != cudaSuccess) { \
//             fprintf(stderr, "CUDA error in %s:%d - %s\n", \
//                    __FILE__, __LINE__, cudaGetErrorString(err)); \
//             exit(EXIT_FAILURE); \
//         } \
//     } while(0)

// // Device functions
// __device__ void d_Wpm(double U, double kx, double ky, double& skf) {
//     double phik = pow(((kx / sqrt(kx * kx + ky * ky + 1e-10)) + 1) / 2, 2);
//     skf = phik / pow((kx * kx + ky * ky + 1e-10), 2) / 2;
// }

// __device__ void d_omega(double kx, double ky, double& omg) {
//     double g = 9.81;
//     double k = kx * kx + ky * ky;
//     omg = sqrt(g * sqrt(k) * (1.0 + k / 363 / 363));
// }

// __device__ void d_dot_product_1(double* a, double* b, int size, double& n) {
//     n = 0;
//     for (int i = 0; i < size; i++) {
//         n += a[i] * b[i];
//     }
// }

// __device__ cuDoubleComplex d_exp_complex(double imag) {
//     return make_cuDoubleComplex(cos(imag), sin(imag));
// }

// // CUDA kernels
// __global__ void kernel_compute_WK(cuDoubleComplex* WK, double UU, double t, 
//                                 int Num, double delt, int Num1) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
    
//     if (i < Num1 && j < Num1) {
//         double k_x = 2 * PI * (i - Num) / Num1 / delt;
//         double k_y = 2 * PI * (j - Num) / Num1 / delt;
//         double wpm, omg;
//         d_Wpm(UU, k_x, k_y, wpm);
//         d_omega(k_x, k_y, omg);
        
//         WK[i * Num1 + j] = cuCmul(make_cuDoubleComplex(wpm, 0.0),
//                                  d_exp_complex(omg * t));
//     }
// }

// __global__ void kernel_compute_WK0(cuDoubleComplex* WK0, double UU,
//                                  int Num, double delt, int Num1) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
    
//     if (i < Num1 && j < Num1) {
//         double k_x = 2 * PI * (i - Num) / Num1 / delt;
//         double k_y = 2 * PI * (j - Num) / Num1 / delt;
//         double wpm;
//         d_Wpm(UU, k_x, k_y, wpm);
//         WK0[i * Num1 + j] = make_cuDoubleComplex(wpm, 0.0);
//     }
// }

// __global__ void kernel_IFT_plus(cuDoubleComplex* input, cuDoubleComplex* output,
//                                int Num, int Num1) {
//     int k = blockIdx.x * blockDim.x + threadIdx.x;
//     if (k < Num1) {
//         cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
//         for (int i = 0; i < Num1; i++) {
//             double phase = -2.0 * PI * i * k / Num1;
//             sum = cuCadd(sum, cuCmul(input[i], d_exp_complex(phase)));
//         }
        
//         double scale = 1.0 / sqrt((double)Num1);
//         int out_idx = (k < Num) ? k : (k - Num + Num);
//         double phase_shift = 2.0 * PI * Num * (out_idx + Num + 1) / Num1;
//         output[out_idx] = cuCmul(make_cuDoubleComplex(scale, 0.0),
//                                 cuCmul(sum, d_exp_complex(phase_shift)));
//     }
// }

// __global__ void kernel_scale_column(cuDoubleComplex* input, cuDoubleComplex* output,
//                                   double scale, int Num1) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < Num1) {
//         output[idx] = cuCmul(input[idx], make_cuDoubleComplex(scale, 0.0));
//     }
// }

// __global__ void kernel_compute_Ker3(cuDoubleComplex* Ker2, cuDoubleComplex* Ker3,
//                                   double* kspk0, double delt, double gt, 
//                                   double the_i, cuDoubleComplex Sum1,
//                                   int Num, int Num1) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
    
//     if (i < Num1 && j < Num1) {
//         double r[2] = { (double)(i - Num), (double)(j - Num) };
//         double kr;
//         d_dot_product_1(kspk0, r, 2, kr);
        
//         double G_xyz = exp(-((i - Num) * delt * (i - Num) * delt + 
//                            (j - Num) * delt * (j - Num) * delt) / 
//                            (gt * gt * pow(cos(the_i), 2)));
        
//         cuDoubleComplex ker2_real = make_cuDoubleComplex(cuCreal(Ker2[i * Num1 + j]), 0.0);
//         cuDoubleComplex exp_term = d_exp_complex(-kr * delt);
//         cuDoubleComplex g_xyz_term = make_cuDoubleComplex(G_xyz, 0.0);
        
//         cuDoubleComplex temp1 = cuCmul(ker2_real, Sum1);
//         cuDoubleComplex temp2 = cuCmul(exp_term, g_xyz_term);
//         Ker3[i * Num1 + j] = cuCmul(temp1, temp2);
//     }
// }

// __global__ void kernel_Int(cuDoubleComplex* Matrix_int, cuDoubleComplex* partial_sums,
//                           double factor, int Num1) {
//     extern __shared__ cuDoubleComplex shared_data[];
//     int tid = threadIdx.x;
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
//     cuDoubleComplex local_sum = make_cuDoubleComplex(0.0, 0.0);
    
//     while (idx < Num1 * Num1) {
//         local_sum = cuCadd(local_sum,
//                           cuCmul(Matrix_int[idx],
//                                 make_cuDoubleComplex(factor, 0.0)));
//         idx += blockDim.x * gridDim.x;
//     }
    
//     shared_data[tid] = local_sum;
//     __syncthreads();
    
//     for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (tid < s) {
//             shared_data[tid] = cuCadd(shared_data[tid], shared_data[tid + s]);
//         }
//         __syncthreads();
//     }
    
//     if (tid == 0) {
//         partial_sums[blockIdx.x] = shared_data[0];
//     }
// }

// // Main function
// void SSA1_linear(complex<double> eps_r, double vk, double the_s, double phi_s,
//     double the_i, double phi_i, double UU, double t, complex<double>* b) {
    
//     double k0x = vk * sin(the_i) * cos(phi_i);
//     double k0y = vk * sin(the_i) * sin(phi_i);
//     double ksx = vk * sin(the_s) * cos(phi_s);
//     double ksy = vk * sin(the_s) * sin(phi_s);
//     double qx = ksx - k0x;
//     double qy = ksy - k0y;
//     double kspk0[2] = { qx, qy };

//     int Num = 200;
//     int Num1 = 2 * Num + 1;
//     double delt = 0.03;
//     double gt = Num1 * delt / 6.0;

//     // Device memory allocation
//     cuDoubleComplex *d_WK, *d_WK0, *d_Ker1, *d_Ker2, *d_Ker3, *d_Sum;
//     cuDoubleComplex *d_row1, *d_row2, *d_col1, *d_col2;
//     double *d_kspk0;
//     size_t size = Num1 * Num1 * sizeof(cuDoubleComplex);
//     size_t vec_size = Num1 * sizeof(cuDoubleComplex);
    
//     CUDA_CHECK(cudaMalloc(&d_WK, size));
//     CUDA_CHECK(cudaMalloc(&d_WK0, size));
//     CUDA_CHECK(cudaMalloc(&d_Ker1, size));
//     CUDA_CHECK(cudaMalloc(&d_Ker2, size));
//     CUDA_CHECK(cudaMalloc(&d_Ker3, size));
//     CUDA_CHECK(cudaMalloc(&d_Sum, 256 * sizeof(cuDoubleComplex)));
//     CUDA_CHECK(cudaMalloc(&d_kspk0, 2 * sizeof(double)));
//     CUDA_CHECK(cudaMalloc(&d_row1, vec_size));
//     CUDA_CHECK(cudaMalloc(&d_row2, vec_size));
//     CUDA_CHECK(cudaMalloc(&d_col1, vec_size));
//     CUDA_CHECK(cudaMalloc(&d_col2, vec_size));
    
//     CUDA_CHECK(cudaMemcpy(d_kspk0, kspk0, 2 * sizeof(double), cudaMemcpyHostToDevice));

//     dim3 blockDim(16, 16);
//     dim3 gridDim((Num1 + blockDim.x - 1) / blockDim.x,
//                  (Num1 + blockDim.y - 1) / blockDim.y);
//     dim3 blockDim1D(256);
//     dim3 gridDim1D((Num1 + blockDim1D.x - 1) / blockDim1D.x);

//     // Compute WK and WK0
//     kernel_compute_WK<<<gridDim, blockDim>>>(d_WK, UU, t, Num, delt, Num1);
//     kernel_compute_WK0<<<gridDim, blockDim>>>(d_WK0, UU, Num, delt, Num1);
//     CUDA_CHECK(cudaDeviceSynchronize());

//     // Compute Sum1
//     int threads = 256;
//     int blocks = min(256, (Num1 * Num1 + threads - 1) / threads);
//     kernel_Int<<<blocks, threads, threads * sizeof(cuDoubleComplex)>>>
//         (d_WK0, d_Sum, pow(2 * PI / Num1 / delt, 2), Num1);
//     CUDA_CHECK(cudaDeviceSynchronize());

//     cuDoubleComplex h_Sum[256];
//     CUDA_CHECK(cudaMemcpy(h_Sum, d_Sum, blocks * sizeof(cuDoubleComplex),
//                          cudaMemcpyDeviceToHost));
//     cuDoubleComplex Sum1 = make_cuDoubleComplex(0.0, 0.0);
//     for (int i = 0; i < blocks; i++) {
//         Sum1 = cuCadd(Sum1, h_Sum[i]);
//     }

//     // Compute Ker1 (row-wise IFT)
//     for (int nrow = 0; nrow < Num1; nrow++) {
//         CUDA_CHECK(cudaMemcpy(d_row1, d_WK + nrow * Num1, vec_size, 
//                             cudaMemcpyDeviceToDevice));
//         kernel_IFT_plus<<<gridDim1D, blockDim1D>>>(d_row1, d_row2, Num, Num1);
//         CUDA_CHECK(cudaDeviceSynchronize());
//         CUDA_CHECK(cudaMemcpy(d_Ker1 + nrow * Num1, d_row2, vec_size,
//                             cudaMemcpyDeviceToDevice));
//     }

//     // Compute Ker2 (column-wise IFT)
//     for (int ncol = 0; ncol < Num1; ncol++) {
//         // Extract column
//         for (int nrow = 0; nrow < Num1; nrow++) {
//             CUDA_CHECK(cudaMemcpy(d_col1 + nrow, d_Ker1 + nrow * Num1 + ncol, 
//                                 sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
//         }
//         kernel_IFT_plus<<<gridDim1D, blockDim1D>>>(d_col1, d_col2, Num, Num1);
//         CUDA_CHECK(cudaDeviceSynchronize());
        
//         // Scale column on device
//         kernel_scale_column<<<gridDim1D, blockDim1D>>>(d_col2, d_col1, 
//                                                       pow(2 * PI / delt, 2), Num1);
//         CUDA_CHECK(cudaDeviceSynchronize());
        
//         // Store back
//         for (int nrow = 0; nrow < Num1; nrow++) {
//             CUDA_CHECK(cudaMemcpy(d_Ker2 + nrow * Num1 + ncol, d_col1 + nrow,
//                                 sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
//         }
//     }

//     // Compute Ker3 and Sum2
//     kernel_compute_Ker3<<<gridDim, blockDim>>>(d_Ker2, d_Ker3, d_kspk0, delt,
//                                               gt, the_i, Sum1, Num, Num1);
//     CUDA_CHECK(cudaDeviceSynchronize());

//     kernel_Int<<<blocks, threads, threads * sizeof(cuDoubleComplex)>>>
//         (d_Ker3, d_Sum, pow(delt, 2), Num1);
//     CUDA_CHECK(cudaDeviceSynchronize());

//     CUDA_CHECK(cudaMemcpy(h_Sum, d_Sum, blocks * sizeof(cuDoubleComplex),
//                          cudaMemcpyDeviceToHost));
//     cuDoubleComplex Sum2 = make_cuDoubleComplex(0.0, 0.0);
//     for (int i = 0; i < blocks; i++) {
//         Sum2 = cuCadd(Sum2, h_Sum[i]);
//     }

//     // Final computation
//     complex<double> factor = 4 * PI * PI / pow(2 * PI, 4);
//     complex<double> Sum2_host(cuCreal(Sum2), cuCimag(Sum2));
//     b[0] = factor * Sum2_host;
//     b[1] = factor * Sum2_host;

//     // Cleanup
//     CUDA_CHECK(cudaFree(d_WK));
//     CUDA_CHECK(cudaFree(d_WK0));
//     CUDA_CHECK(cudaFree(d_Ker1));
//     CUDA_CHECK(cudaFree(d_Ker2));
//     CUDA_CHECK(cudaFree(d_Ker3));
//     CUDA_CHECK(cudaFree(d_Sum));
//     CUDA_CHECK(cudaFree(d_kspk0));
//     CUDA_CHECK(cudaFree(d_row1));
//     CUDA_CHECK(cudaFree(d_row2));
//     CUDA_CHECK(cudaFree(d_col1));
//     CUDA_CHECK(cudaFree(d_col2));
// }

// // Host helper functions
// void Wpm(double U, double kx, double ky, double& skf) {
//     double phik = pow(((kx / sqrt(kx * kx + ky * ky + 1e-10)) + 1) / 2, 2);
//     skf = phik / pow((kx * kx + ky * ky + 1e-10), 2) / 2;
// }

// void omega(double kx, double ky, double& omg) {
//     double g = 9.81;
//     double k = kx * kx + ky * ky;
//     omg = sqrt(g * sqrt(k) * (1.0 + k / 363 / 363));
// }

// void dot_product_1(double* a, double* b, int size, double& n) {
//     n = 0;
//     for (int i = 0; i < size; i++) {
//         n += a[i] * b[i];
//     }
// }


#include "main.h"
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

#define PI          3.1415926535897932384626433832795
#define VC          2.99792458e8 
#define nodetime    200
#define NUM    200

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d - %s\n", \
                   __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Device functions
__device__ void d_Wpm(double U, double kx, double ky, double& skf) {
    double a = 8.1 * pow(10, -3);
    double b = 0.74;
    double g = 9.81;
    double Exp = exp(-b * g * g / (kx * kx + ky * ky + 1e-6) / pow(U, 4));
    double phik = pow(((kx / sqrt(kx * kx + ky * ky + 1e-10)) + 1) / 2, 2);
    //double phik = 1.0 / 4.0 * pow((1 + ky / sqrt(kx * kx + ky * ky + 1e-10)), 2);//侧风
    //double phik = pow((1.0 - kx / sqrt(kx*kx + ky*ky + pow(10, -10))) / 2.0, 2);//顺风
    //double phik = pow(((kx / sqrt(kx*kx + ky*ky + 1e-10)) + 1) / 2, 2);//逆风
    skf = a * Exp * phik / pow((kx * kx + ky * ky + 1e-10), 2) / 2;
    //skf = phik / pow((kx * kx + ky * ky + 1e-10), 2) / 2;
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
    for (int i = 0; i < 2 * NUM + 1; i++)
    {
        for (int j = 0; j < 2 * NUM + 1; j++)
        {
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
                                double UU, double t, int Num, double delt, int Num1) {
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
        //printf("%g  %g\n", k_x, k_y);
        //printf("WK0= %g  %g\n", WK0[idx].x, WK0[idx].y);

    }
}

__global__ void kernel_2D_IFT_and_Ker(cuDoubleComplex* WK, cuDoubleComplex* Ker3,
                                    double* kspk0, double delt, double gt, 
                                    double the_i, double vk, cuDoubleComplex Sum1,
                                    int Num, int Num1) {
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
            //printf("WK= %g %g\n", WK[i * Num1 + col].x, WK[i * Num1 + col].y);
        }
        
        double scale1 = 1.0 / sqrt((double)Num1);
        int out_row = (row < Num) ? row : (row - Num + Num);
    
        double phase_shift1 = 2.0 * PI * Num * (out_row + Num + 1) / Num1;
        cuDoubleComplex ker1 = cuCmul(make_cuDoubleComplex(scale1, 0.0),
                                    cuCmul(sum1, d_exp_complex(phase_shift1)));
        //printf("ker1= %g %g\n", ker1.x, ker1.y);
        // Store in shared memory for second IFT
        shared_data[ty * blockDim.x + tx] = ker1;
        __syncthreads();
        
        // Second IFT (column-wise)
        cuDoubleComplex sum2 = make_cuDoubleComplex(0.0, 0.0);
        for (int j = 0; j < Num1; j++) {
            double phase = -2.0 * PI * j * col / Num1;
            cuDoubleComplex val = (j / blockDim.y == by) ? 
                                 shared_data[(j % blockDim.y) * blockDim.x + tx] : 
                                 make_cuDoubleComplex(0.0, 0.0);
            sum2 = cuCadd(sum2, cuCmul(val, d_exp_complex(phase)));
        }
        
        double scale2 = pow(2 * PI / delt, 2) / sqrt((double)Num1);
        int out_col = (col < Num) ? col : (col - Num + Num);
        double phase_shift2 = 2.0 * PI * Num * (out_col + Num + 1) / Num1;
        cuDoubleComplex ker2 = cuCmul(make_cuDoubleComplex(scale2, 0.0),
                                    cuCmul(sum2, d_exp_complex(phase_shift2)));
        //printf("ker2= %g %g\n", ker2.x, ker2.y);
        // Compute Ker3 directly
        double r[2] = { (double)(out_row - Num), (double)(out_col - Num) };
        double kr;
        double q0= vk * cos(the_i);
        //printf("q0= %g\n", pow(q0 + q0, 2));
        d_dot_product_1(kspk0, r, 2, kr);
        
        double G_xyz = exp(-((out_row - Num) * delt * (out_row - Num) * delt + 
                           (out_col - Num) * delt * (out_col - Num) * delt) / 
                           (gt * gt * pow(cos(the_i), 2)));
        //double T1 = exp(pow(q0 + q0, 2) * cuCreal(ker2));
        //printf("q0= %g\n", T1);
        //cuDoubleComplex ker2_real = make_cuDoubleComplex(T1, 0.0);
        cuDoubleComplex ker2_real = make_cuDoubleComplex(cuCreal(ker2), 0.0);
        //cuDoubleComplex ker2_real = cuCsub(d_exp_complex(pow(q0 + q0, 2)* cuCreal(ker2)), make_cuDoubleComplex(1.0,0.0));
        //cuDoubleComplex ker2_real = cuCsub(T1_term, make_cuDoubleComplex(1.0, 0.0));
        cuDoubleComplex exp_term = d_exp_complex(-kr * delt);
        cuDoubleComplex g_xyz_term = make_cuDoubleComplex(G_xyz, 0.0);
        
        cuDoubleComplex temp1 = cuCmul(ker2_real, Sum1);
        //cuDoubleComplex temp1 = cuCmul(ker2_real, d_exp_complex(-pow(q0 + q0, 2)* cuCreal(Sum1)));
        cuDoubleComplex temp2 = cuCmul(exp_term, g_xyz_term);
        Ker3[out_row * Num1 + out_col] = cuCmul(temp1, temp2);
    }
}

__global__ void kernel_Int(cuDoubleComplex* Matrix_int, cuDoubleComplex* partial_sums,
                          double factor, int Num1) {
    extern __shared__ cuDoubleComplex shared_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    cuDoubleComplex local_sum = make_cuDoubleComplex(0.0, 0.0);
    
    while (idx < Num1 * Num1) {
        local_sum = cuCadd(local_sum,
                          cuCmul(Matrix_int[idx],
                                make_cuDoubleComplex(factor, 0.0)));
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
    double the_i, double phi_i, double UU, double t, complex<double>* b) {
    

    double q0, q01, q1, vw, vc = 2.99792458e8, vr2, qv0, qv1, Pinc, ksk01 = 0, ksk0, qx, qy;
    complex<double> q02, q2, C10, C20, B1vv, B1hh, P_inc, p1p2;
    complex<double>  b1v, eps_g;
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
    double kspk0[2] = { qx, qy };
    vw = 2 * PI * freq;//入射波角频率
    qv0 = vw * vw / vc / vc;

    q0 = vk * cos(the_i); //wc
    q1 = vk * cos(the_s);                       //散射波垂直分量
    q2 = vk * sqrt(eps_r - pow(sin(the_s), 2));//散射波在介质中的垂直分量
    q01 = vk * cos(the_i);                    //入射波垂直分量
    q02 = vk * sqrt(eps_r - pow(sin(the_i), 2));//入射波在介质中的垂直分量

    C10 = (eps_r - 1.0) / (eps_r * q1 + q2) / (eps_r * q01 + q02);//Bvv_前半项
    C20 = (-1.0) * (eps_r - 1.0) / (q1 + q2) / (q01 + q02);//Bhh_前半项

    double ks[2] = { ksx, ksy };
    double k0[2] = { kix, kiy };
    dot_product_1(ks, k0, 2, ksk01);

    ksk0 = ksk01 / (vk * vk * sin(the_i) * sin(the_s));//（ks·k0）/|ks·k0|
    b1v = q2 * q02 * ksk0 - eps_r * vk * vk * sin(the_i) * sin(the_s);//Bvv_后半项

    B1vv = C10 * b1v;
    B1hh = C20 * qv0 * ksk0;
    //std::cout << "B1hh=" << B1hh << endl;
    //system("pause");
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
    double *d_kspk0;
    double* d_KX, * d_KY;
    size_t size = Num1 * Num1 * sizeof(cuDoubleComplex);
    
    CUDA_CHECK(cudaMalloc(&d_WK, size));
    CUDA_CHECK(cudaMalloc(&d_WK0, size));
    CUDA_CHECK(cudaMalloc(&d_Ker3, size));
    CUDA_CHECK(cudaMalloc(&d_Sum, 256 * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_kspk0, 2 * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(d_kspk0, kspk0, 2 * sizeof(double), cudaMemcpyHostToDevice));

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
    kernel_Int<<<blocks, threads, threads * sizeof(cuDoubleComplex)>>>
        (d_WK0, d_Sum, pow(2 * PI / Num1 / delt, 2), Num1);
    CUDA_CHECK(cudaDeviceSynchronize());

    cuDoubleComplex h_Sum[256];
    CUDA_CHECK(cudaMemcpy(h_Sum, d_Sum, blocks * sizeof(cuDoubleComplex),
                         cudaMemcpyDeviceToHost));
    cuDoubleComplex Sum1 = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < blocks; i++) {
        Sum1 = cuCadd(Sum1, h_Sum[i]);
    }
    //std::cout <<"实部=" << Sum1.x << "虚部=" << Sum1.y << endl;
    //system("pause");
    

    // Compute 2D IFT and Ker3 in one kernel
    size_t shared_mem_size = blockDim.x * blockDim.y * sizeof(cuDoubleComplex);
    kernel_2D_IFT_and_Ker<<<gridDim, blockDim, shared_mem_size>>>
        (d_WK, d_Ker3, d_kspk0, delt, gt, the_i, vk, Sum1, Num, Num1);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute Sum2
    kernel_Int<<<blocks, threads, threads * sizeof(cuDoubleComplex)>>>
        (d_Ker3, d_Sum, pow(delt, 2), Num1);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_Sum, d_Sum, blocks * sizeof(cuDoubleComplex),
                         cudaMemcpyDeviceToHost));
    cuDoubleComplex Sum2 = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < blocks; i++) {
        Sum2 = cuCadd(Sum2, h_Sum[i]);
    }
    //std::cout << "实部=" << Sum2.x << "虚部=" << Sum2.y << endl;
    //system("pause");
    // Final computation
    //complex<double> factor = 4 * PI * PI / pow(2 * PI, 4) * 4 * q1 * q01 / (pow(q1 + q01, 2)) * pow(abs(B1vv), 2);
    complex<double> factor = 4 * PI * PI / pow(2 * PI, 4);
    complex<double> Sum2_host(cuCreal(Sum2), cuCimag(Sum2));
    //cout << Sum2_host << endl;
    //system("pause");
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