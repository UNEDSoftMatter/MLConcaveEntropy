/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"
#include <stdio.h>

//--- Function to calculate the average gradients of the offset phi
// of the batch  ---
__global__ void kernel_average_grad_phi(int  batch,
					int*  __restrict__ batch_file,
					int*  __restrict__ batch_N,
					real* __restrict__ grad_phi,
					real* __restrict__ delta_out) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i > 0) return; // One only exit

  //-- We determine the file --
  int file_id = batch_file[batch];

  //-- The gradient is not modified for the first data set (it was initialized as 0) --
  if (file_id != 0) {
    //-- We determine the number of data in the batch --
    int Nbatch_data = batch_N[batch];
    
    real sum = 0.0;
    for (int b = 0; b < Nbatch_data; b++){
      // In each neuron we store as many values as data in the batch. b is the batch.    
      sum += delta_out[b];
    }
    
    grad_phi[file_id] = sum / (real)Nbatch_data;
  }
  
}
