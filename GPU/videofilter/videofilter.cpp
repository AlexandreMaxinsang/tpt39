#include "opencv2/opencv.hpp"
#include <fstream>
#include <iostream> // for standard I/O
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>

#define SHOW
#define PI_ 3.14159265359f
#define STRING_BUFFER_LEN 1024

using namespace cv;
using namespace std;

void print_clbuild_errors(cl_program program, cl_device_id device) {
  cout << "Program Build failed\n";
  size_t length;
  char buffer[2048];
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer),
                        buffer, &length);
  cout << "--- Build log ---\n " << buffer << endl;
  exit(1);
}
unsigned char **read_file(const char *name) {
  size_t size;
  unsigned char **output = (unsigned char **)malloc(sizeof(unsigned char *));
  FILE *fp = fopen(name, "rb");
  if (!fp) {
    printf("no such file:%s", name);
    exit(-1);
  }

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  *output = (unsigned char *)malloc(size);
  if (!*output) {
    fclose(fp);
    printf("mem allocate failure:%s", name);
    exit(-1);
  }

  if (!fread(*output, size, 1, fp))
    printf("failed to read file\n");
  fclose(fp);
  return output;
}
void callback(const char *buffer, size_t length, size_t final,
              void *user_data) {
  fwrite(buffer, 1, length, stdout);
}
void checkError(int status, const char *msg) {
  if (status != CL_SUCCESS)
    printf("%s\n", msg);
}
void randomMemInit(float *data, int size) {
  int i;
  for (i = 0; i < size; ++i) {
    data[i] = rand() / (float)RAND_MAX;
  }
}

// creates a gaussian kernel
float *createGaussianKernel(uint32_t size, float sigma) {
  float *ret;
  uint32_t x, y;
  double center = size / 2;
  float sum = 0;
  // allocate and create the gaussian kernel
  ret = (float *)malloc(sizeof(float) * size * size);
  for (x = 0; x < size; x++) {
    for (y = 0; y < size; y++) {
      ret[y * size + x] =
          exp((((x - center) * (x - center) + (y - center) * (y - center)) /
               (2.0f * sigma * sigma)) *
              -1.0f) /
          (2.0f * PI_ * sigma * sigma);
      sum += ret[y * size + x];
    }
  }
  // normalize
  for (x = 0; x < size * size; x++) {
    ret[x] = ret[x] / sum;
  }
  return ret;
}

int main(int, char **) {

  ///////////////////// opencl setup
  char char_buffer[STRING_BUFFER_LEN];
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_context_properties context_properties[] = {CL_CONTEXT_PLATFORM,
                                                0,
                                                CL_PRINTF_CALLBACK_ARM,
                                                (cl_context_properties)callback,
                                                CL_PRINTF_BUFFERSIZE_ARM,
                                                0x1000,
                                                0};
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;
  //////////////////////

  VideoCapture camera("./bourne.mp4");
  if (!camera.isOpened()) // check if we succeeded
    return -1;

  const string NAME = "./output.avi"; // Form the new name with container
  int ex = static_cast<int>(CV_FOURCC('M', 'J', 'P', 'G'));
  Size S = Size((int)camera.get(CV_CAP_PROP_FRAME_WIDTH), // Acquire input size
                (int)camera.get(CV_CAP_PROP_FRAME_HEIGHT));
  // Size S =Size(1280,720);

  VideoWriter outputVideo; // Open the output
  outputVideo.open(NAME, ex, 25, S, true);

  if (!outputVideo.isOpened()) {
    cout << "Could not open the output video for write: " << NAME << endl;
    return -1;
  }
  time_t start, end;
  double diff, tot;
  int count = 0;
  const char *windowName = "filter"; // Name shown in the GUI window.

#ifdef SHOW
  namedWindow(windowName); // Resizable window, might not work on Windows.
#endif

  clGetPlatformIDs(1, &platform, NULL);
  clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer,
                    NULL);
  printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
  clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN,
                    char_buffer, NULL);
  printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
  clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN,
                    char_buffer, NULL);
  printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

  context_properties[1] = (cl_context_properties)platform;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
  queue = clCreateCommandQueue(context, device, 0, NULL);

  // Read the program
  unsigned char **opencl_program = read_file("gaussian_blur.cl");
  program = clCreateProgramWithSource(context, 1, (const char **)opencl_program,
                                      NULL, NULL);
  if (program == NULL) {
    printf("Program creation failed\n");
    return 1;
  }
  int err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
    print_clbuild_errors(program, device);
  kernel = clCreateKernel(program, "gaussian_blur", NULL);



  while (true) {
    int status;
    Mat cameraFrame, displayframe;
    count = count + 1;
    if (count > 299)
      break;
    camera >> cameraFrame;
    time(&start);

    Mat filterframe = Mat(cameraFrame.size(), CV_8UC3);
    Mat grayframe, edge_x, edge_y, edge;
    cvtColor(cameraFrame, grayframe, CV_BGR2GRAY);


    // image size
    uint32_t imgSize = grayframe.rows * grayframe.cols;
    uint32_t size = 3;

	// create the gaussian Kernel
    float *matrix;
    matrix = createGaussianKernel(size, 1.0f);


    unsigned int mem_size_A = imgSize;
    unsigned int mem_size_B = size * size * sizeof(float);
    unsigned int mem_size_C = imgSize;

    // the hosts iputs
    unsigned char *h_A = grayframe.data;
    float *h_B = matrix;
    unsigned char *h_C =(unsigned char *) malloc(mem_size_C);



    // OpenCL device memory for matrices
    cl_mem d_A;
    cl_mem d_B;
    cl_mem d_C;

    // Create the input and output arrays in device memory for our calculation
    d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, mem_size_A, NULL, &err);
    d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, mem_size_B, NULL, &err);
    d_C = clCreateBuffer(context, CL_MEM_READ_WRITE,mem_size_C, NULL, &err);

    if (!d_A || !d_B || !d_C) {
      printf("Error: Failed to allocate device memory!\n");
      exit(1);
    }

    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[2];
    cl_event kernel_event, finish_event;
    status = clEnqueueWriteBuffer(queue, d_A, CL_FALSE, 0,  mem_size_A, h_A, 0,
                                  NULL, &write_event[0]);
    checkError(status, "Failed to transfer input A");

    status = clEnqueueWriteBuffer(queue, d_B, CL_FALSE, 0,mem_size_B, h_B, 0,
                                  NULL, &write_event[1]);

    checkError(status, "Failed to transfer input B");

    // Set kernel arguments.
    unsigned argi = 0;
    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &d_A);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &d_B);
    checkError(status, "Failed to set argument 2");

    status =
        clSetKernelArg(kernel, argi++, sizeof(int), (void *)&grayframe.cols);
    checkError(status, "Failed to set argument 3");

    status =
        clSetKernelArg(kernel, argi++, sizeof(int), (void *)&grayframe.rows);
    checkError(status, "Failed to set argument 4");

    status = clSetKernelArg(kernel, argi++, sizeof(int), (void *)&size);
    checkError(status, "Failed to set argument 5");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &d_C);
    checkError(status, "Failed to set argument 6");
    ///enqueue the kernel into the OpenCL device for execution
    //the total size of 1 dimension of the work items. Basically the whole image buffer size
    size_t globalWorkItemSize = mem_size_A;
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,&globalWorkItemSize,
									NULL,2, write_event, &kernel_event);

    checkError(status, "Failed to set enqueue kernel");

    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, mem_size_C, h_C,
                                1, &kernel_event, &finish_event);
    checkError(status, "Failed to set output");

    time(&end);
    diff = difftime(end, start);
    printf("GPU took %.8lf seconds to run.\n", diff);

    Mat result(cameraFrame.size(), CV_8UC1, h_C);

    cvtColor(result, displayframe, CV_GRAY2BGR);
    cout << "DISPLAY" << displayframe.size() << endl;
    printf("display frame: %d %d\n", displayframe.rows, displayframe.cols);
    outputVideo << displayframe;

#ifdef SHOW
    imshow(windowName, displayframe);
#endif
    diff = difftime(end, start);
    tot += diff;

    // Release local events

    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
  }

  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);
  clFinish(queue);

  outputVideo.release();
  camera.release();
  // printf("FPS %.2lf .\n", 299.0 / tot);

  return EXIT_SUCCESS;
}
