# cuda-sobel-filter
NRA seminar

1. Download and install CUDA.
2. Download and extract [OpenCV](https://opencv.org/releases) to `C:\`.
3. Set Visual Studio build configuration to Release.
4. Include OpenCV header and lib path in Project properties
- C/C++ → General → Additional Include Directories: `path\to\opencv\include`, eg. for `C:\opencv` set it to `C:\opencv\include`
- Linker → General → Additional Library Directories: `path\to\opencv\lib`, eg. for `C:\opencv` set it to `C:\opencv\lib`
- Linker → Input → Additional Dependencies: add `opencv_world4110.lib`and `opencv_world4110d.lib`
5. Download image you want to use to Project directory `cuda-sobel-filter\SobelFilter\large_image.jpg` that is used as:
```cpp
Mat image = imread("large_image.jpg", IMREAD_COLOR);
```