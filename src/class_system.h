/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "config.h"
#include <cuda_runtime.h>
#include <random>

struct class_system {
  int Nneurons;       // Number of neurons per layer
  int Nhidden;        // Number of hidden layers
  int Nfiles;         // Number of files
  int Niterations;    // Number of iterations
  int initialization; // Initialization option.
                      //  1. From a uniform random distribution in the
                      //     interval (-epsilon, epsilon)
                      //  2. Xavier initialization
  real epsilon;       // epsilon for initialization = 1
  int  N_per_batch;   // Number of data per batch (general)
  int  Nbatches;      // Array with the number of data of each batch
  int* batch_index;   // indices pointing the initial index of data of each batch
  int* batch_file;    // Data file which the batch belongs (between 0 and Nfiles-1)
  int* batch_N;       // Number of data stored in the batches  
  real eta;           // learning rate
  real beta1;         // Parameter of the Adams model
  real beta2;         // Parameter of the Adams model
  real epsilon_adam;  // Parameter of the Adams model
  real* data;         // Data to fit
  size_t Ndata;       // Number of data read from the data file
  int* file_end;      // Set of indices of the final position of each file in the data  
  real** W_hidden;    // Weights of the hidden layers
  real*  W_out;       // Weights of the last hidden layers to output
  real** V_hidden;    // Weights V of the hidden layers (See Input convex neural networks (ICNN))
  real*  V_out;       // Weights V of the last hidden layers to output (See Input convex neural networks (ICNN))
  real** b_hidden;    // Biases of the hidden layers.
  real*  b_out;       // Bias of the last hidden layer to output
  real*  phi;         // Weight or bias phi: constant offset of each file  
  real loss_function; // Loss function in the host
  std::vector<std::string> data_file_names; // Data file names  
  std::mt19937 gen;   // random number generator
  int new_calculation; // 0 if it is new. Different than 0 if it continues another one.


  /********* Subroutines **********/
  int read_input(int&   Nneurons_per_layer,
		 int&   Nhidden_layers,
		 int&   Num_files,
		 int&   Niterations,
		 int&   initialization,
		 real&  epsilon,
		 int&   Number_per_batch,
		 real&  eta_sys,
		 real&  beta1_sys,
		 real&  beta2_sys,
		 real&  eps_adam,
		 int&   freq_loss_function,
		 int&   freq_gnu_file,
		 int&   new_calc,
		 int&   steps_to_initialize,
		 real&  initial_reduction_eta);
  int initialize(int  Num_files,
		 int  Niterations,
		 int  initialization,
		 real epsilon,
		 real eta_sys,
		 real beta1_sys,
		 real beta2_sys,
		 real eps_adam,
		 int  new_calc,
		 real initial_reduction_eta);
  void print_info();
  void initialize_pointers();
  void initialize_data_file_names(int Num_files);    
  void constructor(int Nneurons_per_layer,
		   int Nhidden_layers,
		   int Number_per_batch,
		   int Num_files);
  void destructor();
  int  read_data_file(int Num_files);  
  int  initialize_weights();
  int  read_weights();  
  void initialize_batches();
  void copy_pointers_to_device(real**  k_data,
			       int**   k_file_end,					   
			       real*** k_W_hidden,
			       real**  k_W_out,
			       real*** k_V_hidden,
			       real**  k_V_out,
			       real*** k_b_hidden,
			       real**  k_b_out,
			       real**  k_phi,					   
			       real*** k_z,
			       real*** k_a,
			       int**   k_batch_index,
			       int**   k_batch_N,
			       int**   k_batch_file,
			       real**  k_exit_value,
			       real*** k_delta,
			       real**  k_delta_out,
			       real*** k_grad_W_hidden,
			       real**  k_grad_W_out,
			       real*** k_grad_V_hidden,
			       real**  k_grad_V_out,
			       real*** k_grad_b_hidden,
			       real**  k_grad_b_out,
			       real**  k_grad_phi,
			       real*** k_m_W_hidden,
			       real**  k_m_W_out,
			       real*** k_v_W_hidden,
			       real**  k_v_W_out,
			       real*** k_m_V_hidden,
			       real**  k_m_V_out,
			       real*** k_v_V_hidden,
			       real**  k_v_V_out,
			       real*** k_m_b_hidden,
			       real**  k_m_b_out,
			       real*** k_v_b_hidden,
			       real**  k_v_b_out,
			       real**  k_m_phi,
			       real**  k_v_phi,
			       real**  k_loss_function);  
  void print_loss_function(dim3  numBlocks,
			   dim3  threadsPerBlock,
			   int   batch,
			   real* k_loss_function,
			   real* k_exit_value,
			   int*  k_batch_index,
			   int*  k_batch_N,			   
			   real* k_data,
			   int   step);  
  void free_device_double_pointer(real** k_pointer);
  void pick_batch(int &batch);
  void print_approximation_function(int    step,
				    real** k_W_hidden,
				    real*  k_W_out,
				    real** k_V_hidden,
				    real*  k_V_out,				    
				    real** k_b_hidden,
				    real*  k_b_out,				    
				    real*  k_phi);  
};
