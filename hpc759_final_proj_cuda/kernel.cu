/////////////////////////////////////////////////////////////////////////////
// Semester:         ME759 Fall 2017 
// PROJECT:          GPU Accelerated Edge Detection in Video
// FILE:             kernel.cu
//
// TEAM:    
// Authors: 
// Author1: Nikhil S. Nakhate, nakhate@wisc.edu,
// Author2: Sapan Gupta, sgupta223@wisc.edu
//
//////////////////////////// /////////// //////////////////////////////////

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <chrono>
#include <numeric>

#define batch_size 1
#define delay 1


using namespace cv;
using namespace std;

// device constant memory arrays for convolution kernels
__constant__ int c_sobel_x[3][3];
__constant__ int c_sobel_y[3][3];
__constant__ int c_gaussian[5][5];
const string window_name = "This";

// Structure for device pointers used between the kernel calls
struct dev_mats {
	unsigned char *d_frame, *d_out_gaussian, *d_out_suppress, *d_out_sobel_grad, *d_out_hys_high, *d_out_hys_low;
	int *d_out_sobel_x, *d_out_sobel_y, *d_strong_edge_mask;
}dev_mats1;

// convolution kernels
int sobel_x[3][3] = {
	{ -1, 0, 1 },
	{ -2, 0, 2 },
	{ -1, 0, 1 }
};
int sobel_y[3][3] = {
	{ -1, -2, -1 },
	{ 0, 0, 0 },
	{ 1, 2, 1 }
};
int gaussian[5][5] = {
	{ 2, 4, 5, 4, 2 },
	{ 4, 9, 12, 9, 4 },
	{ 5,  12,  15, 12, 5 },
	{ 4, 9, 12, 9, 4 },
	{ 2, 4, 5, 4, 2 },
};

// This method streams the video frame by frame in batches to the device kernels and renders them using OpenCV APIs
int loadVideo(string path);
// This is the powerhouse method which invokes all the device kernels
void invokeKernel(unsigned char* in_frame, unsigned char* out_frame, struct dev_mats &vec_dev_mat, int rows, int cols, cudaStream_t &stream);


__global__ void applyFiltersGaussian(uchar* d_input, const size_t width, const size_t height, const int kernel_width, uchar* d_output)
{
	extern __shared__ uchar s_input2[];

	//2D Index of current thread
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if ((x<width) && (y<height))
	{
		const int crtShareIndex = threadIdx.y * blockDim.x + threadIdx.x;
		const int crtGlobalIndex = y * width + x;
		s_input2[crtShareIndex] = d_input[crtGlobalIndex];
		__syncthreads();

		const int r = (kernel_width - 1) / 2;
		int sum = 0;

		for (int i = -r; i <= r; ++i)
		{
			int crtY = threadIdx.y + i; //clamp the neighbor pixel, prevent overflow
			if (crtY < 0)						crtY = 0;
			else if (crtY >= blockDim.y)   		crtY = blockDim.y - 1;

			for (int j = -r; j <= r; ++j)
			{
				int crtX = threadIdx.x + j;
				if (crtX < 0) 					crtX = 0;
				else if (crtX >= blockDim.x)	crtX = blockDim.x - 1;

				const float inputPix = (float)(s_input2[crtY * blockDim.x + crtX]);	
				sum += inputPix * c_gaussian[r + j][r + i] / 159;
			}
		}
		d_output[y * width + x] = (uchar)sum;
	}
}


__global__ void applyFiltersSobel(uchar* d_input, const size_t cols, const size_t rows, const int kernel_width,
	int* d_output_x, int* d_output_y, uchar* d_output_grad)
{
	extern __shared__ uchar s_input2[];

	//2D Index of current thread
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if ((x<cols) && (y<rows))
	{
		const int crtShareIndex = threadIdx.y * blockDim.x + threadIdx.x;
		const int crtGlobalIndex = y * cols + x;
		s_input2[crtShareIndex] = d_input[crtGlobalIndex];
		__syncthreads();

		const int r = (kernel_width - 1) / 2;
		int sum_x = 0;
		int sum_y = 0;

		for (int i = -r; i <= r; ++i)
		{
			int crtY = threadIdx.y + i; //clamp the neighbor pixel, prevent overflow
			if (crtY < 0)						crtY = 0;
			else if (crtY >= blockDim.y)   		crtY = blockDim.y - 1;

			for (int j = -r; j <= r; ++j)
			{
				int crtX = threadIdx.x + j;
				if (crtX < 0) 					crtX = 0;
				else if (crtX >= blockDim.x)	crtX = blockDim.x - 1;
				const float inputPix = (float)(s_input2[crtY * blockDim.x + crtX]);
				sum_x += inputPix * c_sobel_x[r + j][r + i];
				sum_y += inputPix * c_sobel_y[r + j][r + i];
			}
		}
		d_output_x[y * cols + x] = sum_x;
		d_output_y[y * cols + x] = sum_y;
		d_output_grad[y * cols + x] = sqrt(pow((double)sum_x, (double)2) + pow((double)sum_y, (double)2));
	}
}


__global__ void cuSuppressNonMax(uchar *mag, int *deltaX, int *deltaY, uchar *nms, int rows, int cols)
{
	const int SUPPRESSED = 0;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= 0 && idx < rows * cols)
	{
		float alpha;
		float mag1, mag2;
		// put zero all boundaries of image
		// TOP edge line of the image
		if ((idx >= 0) && (idx <cols))
			nms[idx] = 0;

		// BOTTOM edge line of image
		else if ((idx >= (rows - 1)*cols) && (idx < (cols * rows)))
			nms[idx] = 0;

		// LEFT & RIGHT edge line
		else if (((idx % cols) == 0) || ((idx % cols) == (cols - 1)))
		{
			nms[idx] = 0;
		}

		else // not the boundaries
		{
			// if magnitude = 0, no edge
			if (mag[idx] == 0)
				nms[idx] = (uchar)SUPPRESSED;
			else {
				if (deltaX[idx] >= 0)
				{
					if (deltaY[idx] >= 0)  // dx >= 0, dy >= 0
					{
						if ((deltaX[idx] - deltaY[idx]) >= 0)       // direction 1 (SEE, South-East-East)
						{
							alpha = (float)deltaY[idx] / deltaX[idx];
							mag1 = (1 - alpha)*mag[idx + 1] + alpha*mag[idx + cols + 1];
							mag2 = (1 - alpha)*mag[idx - 1] + alpha*mag[idx - cols - 1];
						}
						else                                // direction 2 (SSE)
						{
							alpha = (float)deltaX[idx] / deltaY[idx];
							mag1 = (1 - alpha)*mag[idx + cols] + alpha*mag[idx + cols + 1];
							mag2 = (1 - alpha)*mag[idx - cols] + alpha*mag[idx - cols - 1];
						}
					}
					else  // dx >= 0, dy < 0
					{
						if ((deltaX[idx] + deltaY[idx]) >= 0)    // direction 8 (NEE)
						{
							alpha = (float)-deltaY[idx] / deltaX[idx];
							mag1 = (1 - alpha)*mag[idx + 1] + alpha*mag[idx - cols + 1];
							mag2 = (1 - alpha)*mag[idx - 1] + alpha*mag[idx + cols - 1];
						}
						else                                // direction 7 (NNE)
						{
							alpha = (float)deltaX[idx] / -deltaY[idx];
							mag1 = (1 - alpha)*mag[idx + cols] + alpha*mag[idx + cols - 1];
							mag2 = (1 - alpha)*mag[idx - cols] + alpha*mag[idx - cols + 1];
						}
					}
				}

				else
				{
					if (deltaY[idx] >= 0) // dx < 0, dy >= 0
					{
						if ((deltaX[idx] + deltaY[idx]) >= 0)    // direction 3 (SSW)
						{
							alpha = (float)-deltaX[idx] / deltaY[idx];
							mag1 = (1 - alpha)*mag[idx + cols] + alpha*mag[idx + cols - 1];
							mag2 = (1 - alpha)*mag[idx - cols] + alpha*mag[idx - cols + 1];
						}
						else                                // direction 4 (SWW)
						{
							alpha = (float)deltaY[idx] / -deltaX[idx];
							mag1 = (1 - alpha)*mag[idx - 1] + alpha*mag[idx + cols - 1];
							mag2 = (1 - alpha)*mag[idx + 1] + alpha*mag[idx - cols + 1];
						}
					}

					else // dx < 0, dy < 0
					{
						if ((-deltaX[idx] + deltaY[idx]) >= 0)   // direction 5 (NWW)
						{
							alpha = (float)deltaY[idx] / deltaX[idx];
							mag1 = (1 - alpha)*mag[idx - 1] + alpha*mag[idx - cols - 1];
							mag2 = (1 - alpha)*mag[idx + 1] + alpha*mag[idx + cols + 1];
						}
						else                                // direction 6 (NNW)
						{
							alpha = (float)deltaX[idx] / deltaY[idx];
							mag1 = (1 - alpha)*mag[idx - cols] + alpha*mag[idx - cols - 1];
							mag2 = (1 - alpha)*mag[idx + cols] + alpha*mag[idx + cols + 1];
						}
					}
				}

				// non-maximal suppression
				// compare mag1, mag2 and mag[t]
				// if mag[t] is smaller than one of the neighbours then suppress it

				if ((mag[idx] < mag1) || (mag[idx] < mag2))
					nms[idx] = (uchar)SUPPRESSED;
				else
				{
					nms[idx] = (uchar)mag[idx];
				}
			}
		}
	}
}


__global__ void cuHysteresisHigh(uchar *o_frame, uchar *i_frame, int *strong_edge_mask, int t_high, int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < (rows * cols)) {
		/* apply high threshold */
		if (i_frame[idx] > t_high) {
			strong_edge_mask[idx] = 1;
			o_frame[idx] = 255;
		}
		else {
			strong_edge_mask[idx] = 0;
			o_frame[idx] = 0;
		}
	}
}


__device__ void traceNeighbors(uchar *out_pixels, uchar *in_pixels, int idx, int t_low, int cols)
{
	unsigned n, s, e, w;
	unsigned nw, ne, sw, se;

	/* get indices */
	n = idx - cols;
	nw = n - 1;
	ne = n + 1;
	s = idx + cols;
	sw = s - 1;
	se = s + 1;
	w = idx - 1;
	e = idx + 1;

	if (in_pixels[nw] >= t_low) {
		out_pixels[nw] = 255;
	}
	if (in_pixels[n] >= t_low) {
		out_pixels[n] = 255;
	}
	if (in_pixels[ne] >= t_low) {
		out_pixels[ne] = 255;
	}
	if (in_pixels[w] >= t_low) {
		out_pixels[w] = 255;
	}
	if (in_pixels[e] >= t_low) {
		out_pixels[e] = 255;
	}
	if (in_pixels[sw] >= t_low) {
		out_pixels[sw] = 255;
	}
	if (in_pixels[s] >= t_low) {
		out_pixels[s] = 255;
	}
	if (in_pixels[se] >= t_low) {
		out_pixels[se] = 255;
	}
}


__global__ void cuHysteresisLow(uchar *out_pixels, uchar *in_pixels, int *strong_edge_mask, int t_low, int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ((idx > cols)                               
		&& (idx < (rows * cols) - cols) 
		&& ((idx % cols) < (cols - 1))        
		&& ((idx % cols) > (0)))                  
	{
		if (1 == strong_edge_mask[idx]) { 
			traceNeighbors(out_pixels, in_pixels, idx, t_low, cols);
		}
	}
}


void setConvolutionKernelsInDeviceConstantMem() {
	cudaMemcpyToSymbol(c_sobel_x, sobel_x, sizeof(int) * 9);
	cudaMemcpyToSymbol(c_sobel_y, sobel_y, sizeof(int) * 9);
	cudaMemcpyToSymbol(c_gaussian, gaussian, sizeof(int) * 25);
}


int main(int argc, char** argv)
{	
	if (argc <= 1) {
		std::cout << "No CLI argument provided.";
		exit(1);
	}
	// Accepts video path as command line argument
	string path = argv[1];
	
	// Transfers the filter kernels in the device constant memory
	setConvolutionKernelsInDeviceConstantMem();
	
	// Triggers the video processing for edge detection
	loadVideo(path);
	return 0;
}


int loadVideo(string path) {
	cudaStream_t streams[batch_size];
	for (int i = 0; i < batch_size; i++) {
		cudaStreamCreate(&streams[i]);
	}

	VideoCapture capVideo(path);
	if (!capVideo.isOpened()) {
		cout << "Cannot open the video" << endl;
		waitKey(0);
		return -1;
	}

	Size S = Size((int)capVideo.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
		(int)capVideo.get(CV_CAP_PROP_FRAME_HEIGHT));
	int ex = static_cast<int>(capVideo.get(CV_CAP_PROP_FOURCC));

	double fps = capVideo.get(CV_CAP_PROP_FPS);
	double startTime = 1;
	double currentTime = startTime;
	capVideo.set(CV_CAP_PROP_POS_MSEC, startTime);
	//cout << "Frames per second: " << fps << endl;
	

	vector<float> v_fps;
	float count_fps = 0;
	float count_fps1 = 0;
	auto t_start = std::chrono::high_resolution_clock::now();
	auto t_start1 = std::chrono::high_resolution_clock::now();

	vector<Mat> prev_in_frame_q;
	vector<unsigned char*> proc_q;
	vector<dev_mats> vec_dev_mat(batch_size);
	vector<unsigned char*> out_frame_q(batch_size);
	vector<Mat> in_colorframe_q;
	vector<Mat> in_frame_q;

	int flag = 0;
	int frame_empty = 0;
	namedWindow(window_name, CV_WINDOW_NORMAL);
	while (1) {
		if (count_fps == 0.0) {
			t_start = std::chrono::high_resolution_clock::now();
		}
		if (count_fps1 == 0.0) {
			t_start1 = std::chrono::high_resolution_clock::now();
		}

		if (flag == 0) {
			for (int i = 0; i < batch_size; i++) {
				Mat colorframe;
				capVideo >> colorframe;
				if (colorframe.empty()) {
					frame_empty = 1;
					break;
				}
				Mat frame;
				cvtColor(colorframe, frame, COLOR_BGR2GRAY);
				in_colorframe_q.push_back(colorframe);
				in_frame_q.push_back(frame);
				out_frame_q[i] = new unsigned char[frame.rows * frame.cols]();
			}
			if (frame_empty == 1)
				break;
			//cout << "frames collected" << endl;
		}

#pragma omp parallel for
		for (int i = 0; i < batch_size; i++) {
			vector<unsigned char> i_frame(in_frame_q[i].rows*in_frame_q[i].cols);

			if (in_frame_q[i].isContinuous()) {
				i_frame.assign(in_frame_q[i].datastart, in_frame_q[i].dataend);
			}
			else
				continue;

			invokeKernel(&i_frame[0], out_frame_q[i], vec_dev_mat[i], in_frame_q[i].rows, in_frame_q[i].cols, streams[i]);
		}
		//cout << "kernel invoked" << endl;

		if (flag == 1) {
			for (int i = 0; i < batch_size; i++) {
				Mat canvas;
				Mat grayBGR;
				Mat modified_frame(in_frame_q[i].rows, in_frame_q[i].cols, in_frame_q[i].type(), proc_q[i], in_frame_q[i].step);
				cvtColor(modified_frame, grayBGR, COLOR_GRAY2BGR);

				Mat frames[2] = { prev_in_frame_q[i], grayBGR };

				hconcat(frames, 2, canvas);

				imshow(window_name, modified_frame);

				auto t_end = std::chrono::high_resolution_clock::now();
				if (std::chrono::duration<double, std::milli>(t_end - t_start).count() < 5000) {
					count_fps++;
					count_fps1++;
				}
				else {
					v_fps.push_back(count_fps / 5);
					cout << "Frame rate 5s achieved : " << std::accumulate(v_fps.begin(), v_fps.end(), 0.0) / v_fps.size() << endl;
					count_fps = 0;
				}
				if (std::chrono::duration<double, std::milli>(t_end - t_start1).count() > 30000) {
					cout << "Frame rate achieved : " << std::accumulate(v_fps.begin(), v_fps.end(), 0.0) / v_fps.size() << endl;
				}
				else {
					count_fps1 = 0;
				}

				int keyPressed = waitKey(delay);
				if (keyPressed == 27) {
					cout << endl;
					cout << "Frame rate achieved : " << std::accumulate(v_fps.begin(), v_fps.end(), 0.0) / v_fps.size() << endl;
					cout << "ESC pressed" << endl;
					waitKey(0);
					break;
				}
				else if (keyPressed != -1) {
					currentTime = capVideo.get(CV_CAP_PROP_POS_MSEC) + startTime;
					capVideo.set(CV_CAP_PROP_POS_MSEC, currentTime);
					cout << "Pressed=" << keyPressed << " : Forwarding the video by 60s" << endl;
				}
			}
			//cout << "frames rendered" << endl;
		}

		prev_in_frame_q.swap(in_colorframe_q);
		
		in_colorframe_q.clear();
		in_frame_q.clear();

		for (int i = 0; i < batch_size; i++) {
			Mat colorframe;
			capVideo >> colorframe;
			if (colorframe.empty()) {
				break;
			}

			Mat frame;
			cvtColor(colorframe, frame, COLOR_BGR2GRAY);
			in_colorframe_q.push_back(colorframe);
			in_frame_q.push_back(frame);		
		}
		//cout << "frames collected" << endl;
		
		cudaDeviceSynchronize();
		//cout << "synchronized batch" << endl;
		proc_q.swap(out_frame_q);

		out_frame_q.clear();
		out_frame_q.resize(batch_size);

		for (int i = 0; i < batch_size; i++) {
				cudaFree(vec_dev_mat[i].d_frame);			
				cudaFree(vec_dev_mat[i].d_out_gaussian);
				cudaFree(vec_dev_mat[i].d_out_sobel_x);	
				cudaFree(vec_dev_mat[i].d_out_sobel_y);	
				cudaFree(vec_dev_mat[i].d_out_sobel_grad);
				cudaFree(vec_dev_mat[i].d_out_suppress);
				cudaFree(vec_dev_mat[i].d_out_hys_high);	
				cudaFree(vec_dev_mat[i].d_strong_edge_mask);
				cudaFree(vec_dev_mat[i].d_out_hys_low);
				out_frame_q[i] = new unsigned char[in_frame_q[i].rows * in_frame_q[i].cols]();
		}
		
		//cout << "synchronization complete" << endl;

		flag = 1;
		/*Mat modified_frame;
		Canny(frame, modified_frame, 35, 90);*/		
	}

	waitKey(0); // Wait for any keystroke in the window
	destroyWindow("ThisVideo"); //destroy the created window
	cout << "ENDING THIS PROCESS !!";
	return 0;
}


void invokeKernel(unsigned char* in_frame, unsigned char* out_frame, struct dev_mats &vec_dev_mat, int rows, int cols, cudaStream_t &stream) {
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		waitKey(0);
		exit(1);
	}

	// Allocate GPU buffers for two vectors (one input, one output)
	cudaStatus = cudaMalloc((void**)&vec_dev_mat.d_frame, rows*cols * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		waitKey(0);
		exit(1);
	}

	cudaStatus = cudaMalloc((void**)&vec_dev_mat.d_out_gaussian, rows*cols * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		waitKey(0);
		exit(1);
	}

	cudaStatus = cudaMalloc((void**)&vec_dev_mat.d_out_sobel_grad, rows*cols * sizeof(uchar));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		waitKey(0);
		exit(1);
	}

	cudaStatus = cudaMalloc((void**)&vec_dev_mat.d_out_sobel_x, rows*cols * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		waitKey(0);
		exit(1);
	}

	cudaStatus = cudaMalloc((void**)&vec_dev_mat.d_out_sobel_y, rows*cols * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		waitKey(0);
		exit(1);
	}

	cudaStatus = cudaMalloc((void**)&vec_dev_mat.d_out_suppress, rows*cols * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		waitKey(0);
		exit(1);
	}

	cudaStatus = cudaMalloc((void**)&vec_dev_mat.d_out_hys_high, rows*cols * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		waitKey(0);
		exit(1);
	}

	cudaStatus = cudaMalloc((void**)&vec_dev_mat.d_out_hys_low, rows*cols * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		waitKey(0);
		exit(1);
	}

	cudaStatus = cudaMalloc((void**)&vec_dev_mat.d_strong_edge_mask, rows*cols * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		waitKey(0);
		exit(1);
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpyAsync(vec_dev_mat.d_frame, in_frame, rows*cols * sizeof(unsigned char), cudaMemcpyHostToDevice, stream);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		waitKey(0);
		exit(1);
	}

	// cuda kernel execution configurations
	const int threadsPerBlock = 1024;
	const int blocksPerGrid = rows*cols * 1023 / 1024;
	const dim3 blockDimHist(16, 16, 1);
	const dim3 gridDimHist(ceil((float)cols / blockDimHist.x), ceil((float)rows / blockDimHist.y), 1);

	// Launch kernels on the GPU for Canny Edge detection algorithm
	applyFiltersGaussian << <gridDimHist, blockDimHist, blockDimHist.x * blockDimHist.y * sizeof(uchar), stream >> >(vec_dev_mat.d_frame, cols, rows, 5, vec_dev_mat.d_out_gaussian);
	applyFiltersSobel << <gridDimHist, blockDimHist, blockDimHist.x * blockDimHist.y * sizeof(uchar), stream >> >(vec_dev_mat.d_out_gaussian, cols, rows, 3,
																										vec_dev_mat.d_out_sobel_x, vec_dev_mat.d_out_sobel_y, vec_dev_mat.d_out_sobel_grad);
	cuSuppressNonMax << <blocksPerGrid, threadsPerBlock, 0, stream>> > (vec_dev_mat.d_out_sobel_grad, vec_dev_mat.d_out_sobel_x, vec_dev_mat.d_out_sobel_y,
		vec_dev_mat.d_out_suppress, rows, cols);
	cuHysteresisHigh << <blocksPerGrid, threadsPerBlock,0, stream >> > (vec_dev_mat.d_out_hys_high, vec_dev_mat.d_out_suppress, vec_dev_mat.d_strong_edge_mask, 90, rows, cols);
	cuHysteresisLow << <blocksPerGrid, threadsPerBlock,0, stream >> > (vec_dev_mat.d_out_hys_low, vec_dev_mat.d_out_hys_high, vec_dev_mat.d_strong_edge_mask, 35, rows, cols);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		waitKey(0);
		exit(1);
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpyAsync(out_frame, vec_dev_mat.d_out_hys_low, rows*cols * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		waitKey(0);
		exit(1);
	}

}
