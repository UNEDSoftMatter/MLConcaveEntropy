/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "class_system.h"
#include "config.h"
#include <stdio.h>

// Initialization of data batches. Following arrays are initialized here:
// batch_index
// batch_N
// batch_file
void class_system::initialize_batches() {

  //*** Nbatches is defined in system_constructor ****
  
  // batch_index[k] is the first index of the batch k
  int Nbatch_accumulated = 0;
  for (int k = 0; k < Nfiles; k++) {

    //Index where file k starts
    int file_start;
    if (k == 0)
      file_start = 0;
    else
      file_start = file_end[k-1];

    //Number of batches in file k
    int Nbatch_k;
    Nbatch_k = (file_end[k] - file_start + N_per_batch - 1) / N_per_batch;

    //Loop on the batches of file k
    for (int j = 0; j < Nbatch_k; j++) {
      //Index where the batch j of the file k starts
      int batch_j = Nbatch_accumulated + j;
      batch_index[batch_j] = file_start + j * N_per_batch;

      //Number of elements in the batch k of the file k
      if (j < Nbatch_k - 1)
	batch_N[batch_j] = N_per_batch;
      else  // j = Nbatch_k - 1 (last batch of the file, incomplete)
	batch_N[batch_j] = file_end[k] - batch_index[batch_j];
     
      batch_file[batch_j] = k;
    }
    Nbatch_accumulated = Nbatch_accumulated + Nbatch_k;
  }

  
}

