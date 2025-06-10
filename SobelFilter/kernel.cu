#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "device_launch_parameters.h"
using namespace cv;
using namespace std;

// CUDA kernel for grayscale conversion
__global__ void rgb2gray(const uchar3* input, unsigned char* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    uchar3 pixel = input[idx];
    gray[idx] = static_cast<unsigned char>(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z);
}

// CUDA kernel for Sobel filter (X direction only for simplicity)
__global__ void sobelFilter(const unsigned char* gray, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) return;

    int idx = y * width + x;

    int gx = - gray[(y - 1) * width + (x - 1)]
             - 2 * gray[y * width + (x - 1)]
             - gray[(y + 1) * width + (x - 1)]
             + gray[(y - 1) * width + (x + 1)]
             + 2 * gray[y * width + (x + 1)]
             + gray[(y + 1) * width + (x + 1)];

    int gy = - gray[(y - 1) * width + (x - 1)]
             - 2 * gray[(y - 1) * width + x]
             - gray[(y - 1) * width + (x + 1)]
             + gray[(y + 1) * width + (x - 1)]
             + 2 * gray[(y + 1) * width + x]
             + gray[(y + 1) * width + (x + 1)];

    gx = abs(gx);
    gy = abs(gy);

    int magnitude = sqrtf((float)(gx * gx + gy * gy));

    output[idx] = magnitude > 255 ? 255 : magnitude;
}

void sobelFilterCPU(const cv::Mat& gray, cv::Mat& output) {
    int width = gray.cols;
    int height = gray.rows;

    output = cv::Mat::zeros(height, width, CV_8U);

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int gx = - gray.at<uchar>(y - 1, x - 1)
                     - 2 * gray.at<uchar>(y, x - 1)
                     - gray.at<uchar>(y + 1, x - 1)
                     + gray.at<uchar>(y - 1, x + 1)
                     + 2 * gray.at<uchar>(y, x + 1)
                     + gray.at<uchar>(y + 1, x + 1);

            int gy = - gray.at<uchar>(y - 1, x - 1)
                     - 2 * gray.at<uchar>(y - 1, x)
                     - gray.at<uchar>(y - 1, x + 1)
                     + gray.at<uchar>(y + 1, x - 1)
                     + 2 * gray.at<uchar>(y + 1, x)
                     + gray.at<uchar>(y + 1, x + 1);

            int magnitude = sqrtf((float)(gx * gx + gy * gy));

            // saturacija
            output.at<uchar>(y, x) = magnitude > 255 ? 255 : magnitude;
        }
    }
}

int main() {
    // Ucitaj RGB sliku
    Mat image = imread("large_image.jpg", IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Greska: Slika nije ucitana!" << endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    cout << "Dimenzije slike: " << width << "x" << height << endl;

    // ---------- CPU SLOBEL ----------
    Mat gray_cpu, sobel_cpu;
    auto t1 = chrono::high_resolution_clock::now();
    cvtColor(image, gray_cpu, COLOR_BGR2GRAY);
    sobelFilterCPU(gray_cpu, sobel_cpu);
    auto t2 = chrono::high_resolution_clock::now();
    cout << "CPU vrijeme: " << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << " us" << endl;

    // ---------- GPU SOBEL ----------
    uchar3* d_input;
    unsigned char* d_gray, * d_output;
    size_t numPixels = width * height;

    // Alokacija memorije na GPU
    cudaMalloc(&d_input, numPixels * sizeof(uchar3));
    cudaMalloc(&d_gray, numPixels * sizeof(unsigned char));
    cudaMalloc(&d_output, numPixels * sizeof(unsigned char));

    // Kopiraj podatke na GPU
    cudaMemcpy(d_input, image.ptr<uchar3>(), numPixels * sizeof(uchar3), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    auto t3 = chrono::high_resolution_clock::now();
    rgb2gray << <grid, block >> > (d_input, d_gray, width, height);
    cudaDeviceSynchronize();

    sobelFilter << <grid, block >> > (d_gray, d_output, width, height);
    cudaDeviceSynchronize();
    auto t4 = chrono::high_resolution_clock::now();
    cout << "GPU vrijeme: " << chrono::duration_cast<chrono::microseconds>(t4 - t3).count() << " us" << endl;

    // Prebaci rezultat natrag
    Mat result_gpu(height, width, CV_8U);
    cudaMemcpy(result_gpu.ptr(), d_output, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Oslobodi memoriju
    cudaFree(d_input);
    cudaFree(d_gray);
    cudaFree(d_output);

    // Spremi i prikazi rezultate
    imwrite("sobel_cpu.jpg", sobel_cpu);
    imwrite("sobel_gpu.jpg", result_gpu);
    cout << "Rezultati spremljeni." << endl;

    return 0;
}