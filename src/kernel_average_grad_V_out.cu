/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"
#include <stdio.h>

//--- Function to calculate the average gradients of V of the output layer ---
__global__ void kernel_average_grad_V_out(int  batch,
					  int*  __restrict__ batch_index,
					  int*  __restrict__ batch_N,
					  real* __restrict__ grad_V_out,
					  real* __restrict__ delta_out,
					  real* __restrict__ data) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i > 1) return; // Two inputs

  //-- We determine the number of data in the batch --
  int Nbatch_data = batch_N[batch];    
  int row = batch_index[batch];  

  real sum = 0.0;
  for (int b = 0; b < Nbatch_data; b++){
    // The input data es determined
    int start_pos = 3*(row + b);
    real input_data;
    if (i == 0)
      input_data = data[start_pos]; //xdata
    else // i == 1
      input_data = data[start_pos + 1];	  //ydata
    
    // In each neuron we store as many values as data in the batch. b is the batch.    
    sum += delta_out[b] * input_data;
  }
  
  grad_V_out[i] = sum / (real)Nbatch_data;
  
}
