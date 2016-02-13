__kernel void vector_avg(__global const float *input, 
                        __global float *partialSums, 
                        __local float *localSums)
{
	uint local_id = get_local_id(0);
  	uint group_size = get_local_size(0);
	
	localSums[local_id] = input[get_global_id(0)];

    for (uint stride = group_size/2; stride>0; stride /=2)
     {
      // Waiting for each 2x2 addition into given workgroup
      barrier(CLK_LOCAL_MEM_FENCE);

      // Add elements 2 by 2 between local_id and local_id + stride
      if (local_id < stride)
        localSums[local_id] += localSums[local_id + stride];
     }
	if (local_id == 0)
    partialSums[get_group_id(0)] = localSums[0];
}




