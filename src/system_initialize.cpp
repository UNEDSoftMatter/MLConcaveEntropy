/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "class_system.h"
#include "config.h"
#include <math.h>

#include <stdio.h>

// Initialization of the system
int class_system::initialize(int  Num_files,
			     int  Niterations,
			     int  initialization,
			     real epsilon,
			     real eta_sys,
			     real beta1_sys,
			     real beta2_sys,
			     real eps_adam,
			     int  new_calc,
			     real initial_reduction_eta) {

  int error = 0;

  //*** Some variables have been already defined in system_constructor ***//

  this->Nfiles          = Num_files;
  this->Niterations     = Niterations;
  this->initialization  = initialization;
  this->epsilon         = epsilon;
  if (new_calc == 0)
    this->eta             = eta_sys;
  else
    this->eta             = eta_sys / initial_reduction_eta;    
  this->beta1           = beta1_sys;
  this->beta2           = beta2_sys;
  this->epsilon_adam    = eps_adam;
  this->new_calculation = new_calc;

  //--- The random number generator is initialized --
  std::random_device rd;   // This generates the seed
  this->gen.seed(rd());    // Mersenne Twister 19937 (random generator algorithm)  
  // this->gen.seed(12345);   // Fixed seed

  //---- Weights are initialized ----
  if (new_calculation == 0) {
    error = initialize_weights();
    if (error != 0)
      return error;
  }
  else {  // new_calculation != 0 i.e. continuing a previous calculation
      error = read_weights();
    if (error != 0)
      return error;
  }
	 

  //--- Batches are initialized ---
  initialize_batches();

  printf("System initialized\n");

  return 0;
  
}
