/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"

// Activation function of the neural network
__device__ real kernel_activation_function(real x) {
  
  real f;

  //--- Concave non-decreasing function similar to softplus function
  //  f = -log(1.0 + exp(-x));
  // Below a more stable way to write the same function. It is more stable, in case
  // the argument is big and negative (<< -1). This can happen even for normalized inputs, since
  // a_j^k can be greater than 1.
  f = -log1p(exp(-x));
    
  return f;
}
