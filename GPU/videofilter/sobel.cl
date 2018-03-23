__kernel void gaussian_blur(__global const unsigned char *pSrcImage,
                            __global const unsigned char *pDstImage)
{
  unsigned int dstXStride = get_global_size(0);
  unsigned int dstIndex = get_global_id(1)*dstXStride + get_global_id(0);
  unsigned int srcXStride = dstXStride + 32;
  unsigned int srcIndex = get_global_id(1)*srcXStride + get_global_id(0) +16;
  unsigned int a,b,c,d,f,g,h,i;

  a = image[srcIndex-1];
  b = image[srcIndex];
  c = image[srcIndex+1];
  srcIndex += srcXStride;
  d = image[srcIndex-1];
  f = image[srcIndex+1];
  srcIndex += srcXStride;
  g = image[srcIndex-1];
  h = image[srcIndex];
  i = image[srcIndex+1];
  unsigned int xVal = a*1 +d*2 + f*1 + c*(-1) +f*(-2)+i*(-1);
  unsigned int yVal = a*1 +b*2 + c*1 + g*(-1) +h*(-2) +i*(-1);

  pDstImage[dstIndex] = min((unsigned int)255 , (unsigned int)sqrt(xVal*xVal + yVal*yVal));
}
