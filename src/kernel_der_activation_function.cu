/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"

// Derivative of the activation function of the neural network
__device__ real kernel_der_activation_function(real x) {
  
  real der_f;

  //--- Concave non-decreasing function similar to softplus function
  // Note that exp(-x)/(1+exp(-x)) = 1/(1+exp(x)), but the second is
  // more estable (for negative high values << -1)
  der_f = 1.0/(1.0 + exp(x));
  
  return der_f;
}
