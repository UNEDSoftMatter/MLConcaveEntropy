/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"
#include <stdio.h>


//--- Function to calculate the average gradients of weights V
//    of the first hidden layer ---
__global__ void kernel_average_grad_V_hidden(int  batch,
					     int  k,
					     int*   __restrict__ batch_index,
					     int*   __restrict__ batch_N,
					     real** __restrict__ grad_V_hidden,
					     real** __restrict__ delta,
					     real*  __restrict__ data) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= 2 * Nneurons) return; // 2 inputs

  //--- indices ranges:   ---
  // V_0_0 -> 0
  // V_0_1 -> 1
  // ...
  // V_0_Nneurons -> Nneurons - 1
  // V_1_0 -> Nneurons
  // V_1_1 -> Nneurons + 1
  // ...
  // V_1_Nneurons -> 2 * Nneurons - 1
  //---  In general: V_mn -> m * Nneurons + n; here m goes from 0 to Ninputs (2) ----
  
  int n = i % Nneurons;      // destiny neuron
  int m = i / Nneurons;      // origin neuron

  //-- We determine the number of data in the batch --
  int Nbatch_data = batch_N[batch];    
  int row = batch_index[batch];

  real sum = 0.0;
  for (int b = 0; b < Nbatch_data; b++){
    // The input data es determined
    int start_pos = 3*(row + b);
    real input_data;
    if (m == 0)
      input_data = data[start_pos]; //xdata
    else // m == 1
      input_data = data[start_pos + 1];	  //ydata
    
    // In each neuron we store as many values as data in the batch. b is the batch.
    sum += delta[k][n + b*Nneurons] * input_data;
  }
  
  grad_V_hidden[k][i] = sum / (real)Nbatch_data;

}
