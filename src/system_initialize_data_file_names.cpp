/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "class_system.h"
#include "config.h"
#include <stdio.h>

// Data file names are initialized here. Names are data1.dat, data2.dat, etc.
void class_system::initialize_data_file_names(int Num_files) {
  
  for (int i = 0; i < Num_files; ++i) 
    data_file_names.push_back("data" + std::to_string(i+1) + ".dat");
  
}
