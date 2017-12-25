# GPU Accelerated Edge Detection in Video
ME-759 High Performance Computation Final Project

*By: Sapan Gupta, Nikhil S. Nakhate*


--------------------------------------------------------------
## Instructions to build the code

### Requirements:
1. Windows 10
2. Visual studio 2017 
3. CUDA toolkit 9.0.176
4. CUDA driver 388.16
5. OpenCV 3.3.1

### Environment:
* OPENCV_DIR = <opencv/build>
* CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0
* PATH = %OPENCV_DIR%\x64\vc14\lib;%OPENCV_DIR%\x64\vc14\bin;%PATH%


### Instructions to execute the executable:
* $ cd x64\Release
* $ hpc759_final_proj_cuda.exe "The Secret Life Of Walter Mitty.mp4"

### Executable process description:
This will open two windows for the live original and the processed edge contour feeds.
In the edge contour feed, just the edges of the structures and shapes from the original video will get highlighted as white pixels and other areas will be removed as black pixels.
