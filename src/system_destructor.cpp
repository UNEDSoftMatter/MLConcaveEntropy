/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "class_system.h"
#include <stdio.h>

// Destructor of the class system
void class_system::destructor() {
  if (data)
    delete[] data;
  if (file_end)
    delete[] file_end;
  
  for (int k = 0; k < Nhidden; k++) {
    if (W_hidden[k])
      delete[] W_hidden[k];
    if (V_hidden[k])
      delete[] V_hidden[k];
    if (b_hidden[k])
      delete[] b_hidden[k];
  }
  if (W_hidden)
    delete[] W_hidden;
  if (b_hidden)
    delete[] b_hidden;
  
  // output ewights and biases are destroyed
  if (W_out)
    delete[] W_out;
  if (V_out)
    delete[] V_out;    
  if (b_out)
    delete[] b_out;
  if (phi)
    delete[] phi;  

  // Batch indices
  if (batch_index)
    delete[] batch_index;

  // Batch file
  if (batch_file)
    delete[] batch_file;

  // Batch N
  if (batch_N)
    delete[] batch_N;    

  printf("System destroyed\n");
}
