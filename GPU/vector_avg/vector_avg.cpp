#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <iostream> // for standard I/O
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define STRING_BUFFER_LEN 1024
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

// Randomly generate a floating-point number between -10 and 10.
float rand_float() { return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f; }

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
  const unsigned N = 51200000;
  float *input_a = (float *)malloc(sizeof(float) * N);
  cl_mem input_a_buf; // num_devices elements
  cl_mem output_buf;  // num_devices elements
  int status;

  time_t start, end;
  double diff;
  time(&start);
  float avg = 0;
  for (unsigned j = 0; j < N; ++j) {
    input_a[j] = rand_float();
    avg += input_a[j];
    // printf("ref %f\n",ref_output[j]);
  }
  avg /= N;
  time(&end);
  diff = difftime(end, start);
  printf("CPU took %.2lf seconds to run.\n", diff);

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
  size_t max_work_group_size;
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                  &max_work_group_size, NULL);
  size_t group_num = N / max_work_group_size;
  float *output = (float *)malloc(sizeof(float) * group_num);
  for (unsigned j = 0; j < group_num; ++j) {
    output[j] = 0;
  }

  unsigned char **opencl_program = read_file("vector_avg.cl");
  program = clCreateProgramWithSource(context, 1, (const char **)opencl_program,
                                      NULL, NULL);
  if (program == NULL) {
    printf("Program creation failed\n");
    return 1;
  }
  int success = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (success != CL_SUCCESS)
    print_clbuild_errors(program, device);
  kernel = clCreateKernel(program, "vector_avg", NULL);
  // Input buffers.
  input_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float),
                               NULL, &status);
  checkError(status, "Failed to create buffer for input A");

  // Output buffer.
  output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                              group_num * sizeof(float), NULL, &status);
  checkError(status, "Failed to create buffer for output");

  // Transfer inputs to each device. Each of the host buffers supplied to
  // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
  // for the host-to-device transfer.
  cl_event write_event[1];
  cl_event kernel_event, finish_event;
  status =
      clEnqueueWriteBuffer(queue, input_a_buf, CL_FALSE, 0, N * sizeof(float),
                           input_a, 0, NULL, &write_event[0]);
  checkError(status, "Failed to transfer input A");

  // Set kernel arguments.
  unsigned argi = 0;

  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_a_buf);
  checkError(status, "Failed to set argument 1");

  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
  checkError(status, "Failed to set argument 2");

  status = clSetKernelArg(kernel, argi++, group_num * sizeof(float), NULL);
  checkError(status, "Failed to set argument 3");

  const size_t global_work_size = N;
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size,
                                  &max_work_group_size, 1, write_event,
                                  &kernel_event);
  checkError(status, "Failed to launch kernel");
  // Read the result. This the final operation.
  status = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0,
                               group_num * sizeof(float), output, 1,
                               &kernel_event, &finish_event);
  checkError(status, "Failed to set output");
  time(&end);
  diff = difftime(end, start);
  printf("GPU took %.8lf seconds to run.\n", diff);

  // Checking

  float mean = 0;
  for (unsigned j = 0; j < group_num; ++j) {
    mean += output[j];
  }
  mean /= N;
  if (fabsf(mean - avg) > 1.0e-5f) {
    printf("%d", max_work_group_size);
    printf("wrong %f %f \n", mean, avg);
    printf("%f %f \n", output[0], output[1]);
  }

  // Release local events.
  clReleaseEvent(write_event[0]);
  clReleaseKernel(kernel);

  clReleaseCommandQueue(queue);
  clReleaseMemObject(input_a_buf);
  clReleaseMemObject(output_buf);
  clReleaseProgram(program);
  clReleaseContext(context);

  //--------------------------------------------------------------------

  clFinish(queue);

  return 0;
}
