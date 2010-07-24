#include <math.h>
#include <stdio.h>
#include <fftw3.h>

void write_data(FILE *file, fftw_complex *data, int N) {
  int j;
  for (j = 0; j < N; ++j) 
    fprintf(file, "%10.5e %10.5e\n", data[j][0], data[j][1]);
}

int main() {
  fftw_complex *in, *out;
  fftw_plan plan;
  int N = 100;
  double a_in[5] = {0.,0.1,1.0,2.3,1.2};

  in = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N);
  out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N);
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
