#include <math.h>
#include <stdio.h>
#include <fftw3.h>

#define index(i,j,k) (k+N_k*(j+N_j*i) )

typedef struct {
  int N_i, N_j, N_k;
  char *name;
  float *xspace;
  fftw_complex *kspace;
} field;

void write_data(FILE *file, fftw_complex *data, int N) {
  int j;
  for (j = 0; j < N; ++j) 
    fprintf(file, "%10.5e %10.5e\n", data[j][0], data[j][1]);
}

void tg_setup(fftw_complex *vx, fftw_complex *vy, fftw_complex *vz, int k0) {
  /* Initialize a Taylor Green vortex in k space */

  return;
}

field *create_field(char *name, int N_i, int N_j, int N_k) {
  field *new_field;

  new_field = (field *) malloc(size_of(field));
  new_field->name = name;
  new_field->N_i = N_i;
  new_field->N_j = N_j;
  new_field->N_k = N_k;

  return new_field;
}

void init_field(field *new_field) {
  /* Allocate memory for the real and k space data */
  new_field->kspace = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * new_field->N_i
                                                   * new_field->N_j * new_field->N_k);
  new_field->xspace = (double *) malloc(sizeof(double) * new_field->N_i
                                        * new_field->N_j * new_field->N_k);

  return;
}

int main() {
  double *xspace;
  fftw_complex *kspace, *in, *out;
  fftw_plan plan;
  int N = 100;
  int N_i = 100, N_j = 100, N_k = 100;
  double a_in[5] = {0.,0.1,1.0,2.3,1.2};
  //plan = fftw_plan_r2c_3d(N_i, , in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

  // initialize data
  int i,j;
  for (j = 0; j < 5; ++j) {
    for (i = 0; i < N; ++i) {
      float x = 2*M_PI/N * i;
      in[i][0] += a_in[j]*sin(j*x);
      in[i][1] = 0.;
    }
  }

  fftw_execute(plan);

  FILE *output;
  output = fopen("fwd.dat","w");
  write_data(output,out,N);
  close(output);

  // clean up yer mess
  fftw_destroy_plan(plan);
  fftw_free(in);
  fftw_free(out);
  return 0;

}
