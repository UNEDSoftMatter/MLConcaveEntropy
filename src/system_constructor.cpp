/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "class_system.h"
#include "config.h"
#include <stdio.h>
#include <iostream>

// Constructor of the class system
void class_system::constructor(int Nneurons_per_layer,
			       int Nhidden_layers,
			       int Number_per_batch,
			       int Num_files) {

  this->Nneurons       = Nneurons_per_layer;
  this->Nhidden        = Nhidden_layers;

  //**** data array is constructed in system_read_data_file *****

  //-- Arrays for hidden layers and biases are created --
  //-- We save memory for the arrays --  
  W_hidden = new real*[this->Nhidden];
  V_hidden = new real*[this->Nhidden];  
  b_hidden = new real*[this->Nhidden];
  
  //-- Weights between the input and the first hidden layer (2 entries)
  // z is interpreted as z[k][ i + j*Nneurons ]; j is the index of the data in the batch
  W_hidden[0] = new real[this->Nneurons * 2];
  V_hidden[0] = new real[this->Nneurons * 2];  
  b_hidden[0] = new real[this->Nneurons];
  //-- Rest of hidden layers --
  for (int k = 1; k < this->Nhidden; k++) {
    W_hidden[k] = new real[this->Nneurons * this->Nneurons];    
    b_hidden[k] = new real[this->Nneurons];
    V_hidden[k] = new real[this->Nneurons * 2];    
  }

  //-- Weights from last hidden layer to output (1 output) --
  W_out = new real[1 * this->Nneurons];
  V_out = new real[2];  
  b_out = new real[1];
  phi   = new real[Num_files];

  //-- Arrays for batches --
  //-- To use the final incomplete batch of each file--
  this->N_per_batch = Number_per_batch;
  this->Nbatches = 0;
  for (int k = 0; k < Num_files; k++) {
    if (k == 0) {
      this->Nbatches = this->Nbatches +
	(file_end[k] + N_per_batch - 1) / N_per_batch;
    }
    else {   // k > 0
      this->Nbatches = this->Nbatches +
	(file_end[k] - file_end[k-1] + N_per_batch - 1) / N_per_batch;
    }
  }
  batch_index  = new int[Nbatches];
  batch_file   = new int[Nbatches];
  batch_N      = new int[Nbatches];

  printf("Pointers constructed\n");

}
