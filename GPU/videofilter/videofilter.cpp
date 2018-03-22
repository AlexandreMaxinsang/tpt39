#include "opencv2/opencv.hpp"
#include <fstream>
#include <iostream> // for standard I/O
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <CL/cl.h>
#include <CL/cl_ext.h>

using namespace cv;
using namespace std;
#define SHOW

#define WA 1024
#define HA 1024
#define WB 1024

#define HB WA
#define WC WB
#define HC HA

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
  ret = malloc(sizeof(float) * size * size);
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

  while (true) {

    Mat cameraFrame, displayframe;
    count = count + 1;
    if (count > 299)
      break;
    camera >> cameraFrame;
    time(&start);

    Bitmap bmp = null;
    Mat filterframe = Mat(cameraFrame.size(), CV_8UC3);
    Mat grayframe, edge_x, edge_y, edge;
    cvtColor(cameraFrame, grayframe, CV_BGR2GRAY);

    bmp = Bitmap.createBitmap(grayframe.cols(), grayframe.rows(), Bitmap.Config.ARGB_8888);
    Utils.matToBitmap(grayframe, bmp);
    uint32_t imgSize;
    imgSize = bmp.imgWidth*bmp.imgHeight*3;

    //create the gaussian kernel
    matrix = createGaussianKernel(3,0);
    //create the pointer that will hold the new (blurred) image data
    unsigned char* newData;
    newData = malloc(imgSize);

    //--------------------Initializing cl-------------------------

    char char_buffer[STRING_BUFFER_LEN];
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_context_properties context_properties[] = {
        CL_CONTEXT_PLATFORM,
        0,
        CL_PRINTF_CALLBACK_ARM,
        (cl_context_properties)callback,
        CL_PRINTF_BUFFERSIZE_ARM,
        0x1000,
        0};
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    //-------------------------------------------

    h_A = bmp.imgData;
    h_B = matrix;
    h_C = newData;

    mem_size_A = imgSize;
    mem_size_B = 3*3*sizeof(float);
    mem_size_C = imgSize;

    // OpenCL device memory for matrices
    cl_mem d_A;
    cl_mem d_B;
    cl_mem d_C;

    clGetPlatformIDs(1, &platform, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN,
                      char_buffer, NULL);
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
    unsigned char **opencl_program = read_file("kernel.cl");
    program = clCreateProgramWithSource(context, 1, (const char **)opencl_program,
                                        NULL, NULL);
    if (program == NULL) {
      printf("Program creation failed\n");
      return 1;
    }
    int err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
      print_clbuild_errors(program, device);
    kernel = clCreateKernel(program, "kernel", NULL);

    // Create the input and output arrays in device memory for our calculation
    d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_A, NULL, &err);
    d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                         mem_size_A, h_A, &err);
    d_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                         mem_size_B, h_B, &err);

    if (!d_A || !d_B || !d_C) {
      printf("Error: Failed to allocate device memory!\n");
      exit(1);
    }

    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[2];
    // cl_event kernel_event,finish_event;
    status = clEnqueueWriteBuffer(queue, d_A, CL_FALSE, 0, mem_size_A, h_A, 0,
                                  NULL, &write_event[0]);
    checkError(status, "Failed to transfer input A");
    status = clEnqueueWriteBuffer(queue, d_B, CL_FALSE, 0, mem_size_A, h_B, 0,
                                  NULL, &write_event[1]);
    checkError(status, "Failed to transfer input B");

    // Set kernel arguments.
    unsigned argi = 0;
    int wA = WA;
    int wC = WC;

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &d_C);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &d_A);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &d_B);
    checkError(status, "Failed to set argument 3");

    status = clSetKernelArg(kernel, argi++, sizeof(int), (void *)&wA);
    checkError(status, "Failed to set argument 4");

    status = clSetKernelArg(kernel, argi++, sizeof(int), (void *)&wC);
    checkError(status, "Failed to set argument 5");

    size_t localWorkSize[2], globalWorkSize[2];

    localWorkSize[0] = 16;
    localWorkSize[1] = 16;
    globalWorkSize[0] = 1024;
    globalWorkSize[1] = 1024;

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize,
                                 localWorkSize, 0, NULL, NULL);

    if (err != CL_SUCCESS) {
      printf("Error: Failed to execute kernel! %d\n", err);
      exit(1);
    }

    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, mem_size_C, h_C, 0, NULL,
                                 NULL);
    checkError(status, "Failed to set output");

    // Release local events.);
    free(h_A);
    free(h_B);
    free(h_C);
    free(ref_C);

    clReleaseEvent(write_event[0]);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseProgram(program);
    clReleaseContext(context);

    //--------------------------------------------------------------------

    clFinish(queue);


    // GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);
    // GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);
    // GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);
    // Scharr(grayframe, edge_x, CV_8U, 0, 1, 1, 0, BORDER_DEFAULT);
    // Scharr(grayframe, edge_y, CV_8U, 1, 0, 1, 0, BORDER_DEFAULT);
    // addWeighted(edge_x, 0.5, edge_y, 0.5, 0, edge);

    time(&end);

    cvtColor(edge, displayframe, CV_GRAY2BGR);
    outputVideo << displayframe;
#ifdef SHOW
    imshow(windowName, displayframe);
#endif
    diff = difftime(end, start);
    tot += diff;
  }
  outputVideo.release();
  camera.release();
  printf("FPS %.2lf .\n", 299.0 / tot);

  return EXIT_SUCCESS;
}
