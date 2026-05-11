/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "class_system.h"
#include <iostream>
#include <fstream>
#include <vector>
#include "config.h"

// Function to read data from the file
// To access the file of i row and j column:
//     data[i*3 + j]
// row 0 → data[0], data[1], data[2]
// row 1 → data[3], data[4], data[5]
// row 2 → data[6], data[7], data[8]
// ...
// row i → data[i*3 + 0], data[i*3 + 1], data[i*3 + 2]
int class_system::read_data_file(int Num_files) {

  std::vector<real> temp;
  std::vector<int>  temp_file_end;

  for (int n = 0; n < Num_files; n++) {

    std::ifstream file(data_file_names[n]);
    if (!file.is_open()) {
      printf("System read data file error: cannot open file %s.\n", data_file_names[n].c_str());
      return 1;
    }

    real a, b, c;
    while (file >> a >> b >> c) {
        temp.push_back(a);
        temp.push_back(b);
        temp.push_back(c);
    }

    // number of accumulated data until this file
    temp_file_end.push_back(temp.size() / 3);

    file.close();
  }

  // Número total de datos
  Ndata = temp.size() / 3;
  printf("ddd %d\n", Ndata);

  delete[] data;
  delete[] file_end;  
  
  // Dynamic memory is reserved
  data = new real[temp.size()];
  
  // Data is copied to the pointer
  for (size_t i = 0; i < temp.size(); ++i)
    data[i] = temp[i];
  
  // Índices of end files
  file_end = new int[Num_files];
  for (int i = 0; i < Num_files; ++i)
    file_end[i] = temp_file_end[i];

  printf("All data files read. Total data points = %d\n", Ndata);

  return 0;
}


