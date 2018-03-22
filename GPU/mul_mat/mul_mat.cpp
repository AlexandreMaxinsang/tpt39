#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <iostream> // for standard I/O
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define STRING_BUFFER_LEN 1024

#define WA 1024
#define HA 1024
#define WB 1024

#define HB WA
#define WC WB
#define HC HA

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

int main() {
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

  //--------------------------------------------------------------------

  // OpenCL device memory for matrices
  cl_mem d_A;
  cl_mem d_B;
  cl_mem d_C;

  srand(2014);

  // Allocate host memory for matrices A and B
  unsigned int size_A = WA * HA;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float *h_A = (float *)malloc(mem_size_A);

  unsigned int size_B = WB * HB;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B = (float *)malloc(mem_size_B);

  // Initialize host memory
  randomMemInit(h_A, size_A);
  randomMemInit(h_B, size_B);

  // Allocate host memory for the result C
  unsigned int size_C = WC * HC;
  unsigned int mem_size_C = sizeof(float) * size_C;
  float *h_C = (float *)malloc(mem_size_C);

  // Allocate host memory for the result reference C
  float *ref_C = (float *)malloc(mem_size_C);

  int status;

  // Showing the time that just CPU takes for the operation
  time_t start, end;
  double diff;

  time(&start);

  for (int tx = 0; tx < WC; tx++) {
    for (int ty = 0; ty < HC; ty++) {
      float value = 0;
      for (int k = 0; k < WA; ++k) {
        float elementA = h_A[ty * WA + k];
        float elementB = h_B[k * WB + tx];
        value += elementA * elementB;
      }

      // Write the matrix to device memory each
      // thread writes one element
      ref_C[ty * WA + tx] = value;
    }
  }

  time(&end);
  diff = difftime(end, start);
  printf("CPU took %.2lf seconds to run.\n", diff);

  // Showing the time that CPU/GPU take for the operation
  time(&start);
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

  //     size_t max_work_group_size;
  //     clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
  //     &max_work_group_size, NULL);
  //    size_t group_num = N / max_work_group_size;

  //    float *output=(float *) malloc(sizeof(float)* group_num);
  //     for (unsigned j=0; j< group_num; ++j){
  //    output[j] = 0;
  //    }

  // Read the program
  unsigned char **opencl_program = read_file("mul_mat.cl");
  program = clCreateProgramWithSource(context, 1, (const char **)opencl_program,
                                      NULL, NULL);
  if (program == NULL) {
    printf("Program creation failed\n");
    return 1;
  }
  int err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
    print_clbuild_errors(program, device);
  kernel = clCreateKernel(program, "mul_mat", NULL);

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
  checkError(status, "Failed to transfer input A");

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

  time(&end);
  diff = difftime(end, start);
  printf("GPU took %.8lf seconds to run.\n", diff);

  // Checking

  for (int j = 0; j < WC * HC; j++) {
    if (fabsf(h_C[j] - ref_C[j]) > 1.0e-5f) {
      printf("wrong %f %f \n", h_C[j], ref_C[j]);
    }
  }

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

  return 0;
}
