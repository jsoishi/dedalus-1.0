#include <math.h>
#include <stdio.h>
#include <fftw3.h>
#include <stdlib.h>
#define index(i,j) (j+N_j*i)

typedef struct {
  int N_i, N_j, N_ik, N_jk;
  char *name;
  /* double *xspace; */
  fftw_complex *xspace;
  fftw_complex *kspace;
} field;

void write_data(FILE *file, fftw_complex *data, int N) {
  int j;
  for (j = 0; j < N; ++j) 
    fprintf(file, "%10.5e %10.5e\n", data[j][0], data[j][1]);
}

void write_field_xspace(FILE *file, field *field) {
  int i,j, N_j;
  N_j = field->N_j;
  for (j = 0; j < field->N_j; j++) {
    for (i = 0; i < field->N_i; i++) { 
      fprintf(file, "%10.5e\n", field->xspace[index(i,j)][0]);
    }
  }
}

void write_field_kspace(FILE *file, field *field) {
  int i,j, N_j;
  N_j = field->N_j;
  for (j = 0; j < N_j; j++) {
    for (i = 0; i < field->N_i; i++) { 
      fprintf(file, "%10.5e %10.5e\n", field->kspace[index(i,j)][0],field->kspace[index(i,j)][1]);
    }
  }
}

void tg_setup_2d(field *vx, field *vy) {
  /* Initialize a Taylor Green vortex in k space */
  int kx0, kx1, ky0, ky1, N_j;

  N_j = vx->N_j; /* bad hack!!!! */
  kx0 = 1;
  kx1 = vx->N_i -1;
  
  ky0 = 1;
  ky1 = N_j -1;
  vx->kspace[index(kx0,ky0)][0]  = 0.;
  vx->kspace[index(kx0,ky0)][1]  = -1./4.;
  vx->kspace[index(kx1,ky1)][0]  = 0.;
  vx->kspace[index(kx1,ky1)][1]  = 1./4.;
  vx->kspace[index(kx1,ky0)][0]  = 0.;
  vx->kspace[index(kx1,ky0)][1]  = -1./4.;
  vx->kspace[index(kx0,kx1)][0]  = 0.;
  vx->kspace[index(kx0,kx1)][1]  = 1./4.;

  vy->kspace[index(kx0,ky0)][0]  = 0.;
  vy->kspace[index(kx0,ky0)][1]  = 1./4.;
  vy->kspace[index(kx1,ky1)][0]  = 0.;
  vy->kspace[index(kx1,ky1)][1]  = -1./4.;
  vy->kspace[index(kx1,ky0)][0]  = 0.;
  vy->kspace[index(kx1,ky0)][1]  = -1./4.;
  vy->kspace[index(kx0,kx1)][0]  = 0.;
  vy->kspace[index(kx0,kx1)][1]  = 1./4.;
    
  return;
}

field *create_field(char *name, int N_i, int N_j) {
  field *new_field;

  new_field = (field *) malloc(sizeof(field));
  new_field->name = name;
  new_field->N_i = N_i;
  new_field->N_j = N_j;

  return new_field;
}

void init_field(field *new_field) {
  printf("Initializing field %s with N_i = %i, N_j = %i...\n",new_field->name,new_field->N_i, new_field->N_j);
  /* Allocate memory for the real and k space data */
  new_field->kspace = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * new_field->N_i
                                                   * new_field->N_j);
  /* new_field->xspace = (double *) malloc(sizeof(double) * new_field->N_i */
  /*                                       * new_field->N_j); */
  new_field->xspace = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * new_field->N_i
                                                   * new_field->N_j);

  return;
}

int main() {
  int N_i = 100, N_j = 100;

  field *vx, *vy;
  fftw_plan vx_plan;
  vx = create_field("x-velocity",N_i,N_j);
  vy = create_field("y-velocity",N_i,N_j);

  init_field(vx);
  init_field(vy);
  tg_setup_2d(vx,vy);
  vx_plan = fftw_plan_dft_2d(vx->N_i, vx->N_j, vx->kspace, vx->xspace, FFTW_FORWARD, FFTW_ESTIMATE);
  FILE *koutput;
  koutput = fopen("vx_tg_kspace.dat","w");

  write_field_kspace(koutput,vx);
  close(koutput);
  fftw_execute(vx_plan);

  FILE *output;
  output = fopen("vx_tg_real.dat","w");
  write_field_xspace(output,vx);
  close(output);

  fftw_destroy_plan(vx_plan);
  return 0;

}
