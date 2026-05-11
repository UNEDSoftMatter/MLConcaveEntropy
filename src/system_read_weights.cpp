/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/
#include "class_system.h"
#include "config.h"
#include <cstdio>
#include <cstring>

int class_system::read_weights() {

  FILE *file = fopen("weights.gnu", "r");
  if (!file) {
    perror("System read weights error: opening file error");
    return 1;
  }

  char line[512];

  while (fgets(line, sizeof(line), file)) {

    // ---- W weights ----
    if (strncmp(line, "W_", 2) == 0) {
      int n, m, k;
      real value;
      if (sscanf(line, "W_%d_%d_%d = %lf", &n, &m, &k, &value) == 4) {
        if (k < Nhidden) {
          W_hidden[k][n + m * Nneurons] = value;
        } else if (k == Nhidden) {
          W_out[m] = value;
        }
      }
    }

    // ---- V weights ----
    else if (strncmp(line, "V_", 2) == 0) {
      int n, m, k;
      real value;
      if (sscanf(line, "V_%d_%d_%d = %lf", &n, &m, &k, &value) == 4) {
        if (k < Nhidden) {
          V_hidden[k][n + m * Nneurons] = value;
        } else if (k == Nhidden) {
          V_out[m] = value;
        }
      }
    }

    // ---- Biases ----
    else if (strncmp(line, "b_", 2) == 0) {
      int n, k;
      real value;
      if (sscanf(line, "b_%d_%d = %lf", &n, &k, &value) == 3) {
        if (k < Nhidden) {
          b_hidden[k][n] = value;
        } else if (k == Nhidden) {
          b_out[0] = value;
        }
      }
    }

    // ---- Phi offsets ----
    else if (strncmp(line, "phi_", 4) == 0) {
      int i;
      real value;
      if (sscanf(line, "phi_%d = %lf", &i, &value) == 2) {
        phi[i] = value;
      }
    }
  }

  fclose(file);
  printf("Weights loaded from weights.gnu file\n");
  return 0;
}
