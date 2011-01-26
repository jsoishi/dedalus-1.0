#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <fftw3.h>


/* this isn't really right. but it will work for now if you define a local variable called N_j... */
#define index(i,j) (j+N_j*i)
#define GOOD_STEP 0
#define BAD_STEP  1

int debug = 0;

typedef struct {
  int N_i, N_j, N_ik, N_jk;
  char *name;
  /* double *xspace; */
  fftw_complex *xspace;
  fftw_complex *kspace;
  fftw_plan fwd_plan, rev_plan;
  double *kx;
  double *ky;
} field;

void write_data(FILE *file, fftw_complex *data, int N) {
  int j;
  for (j = 0; j < N; ++j) 
    fprintf(file, "%10.5e %10.5e\n", data[j][0], data[j][1]);
}

void write_field_xspace(FILE *file, field *field) {
  fftw_execute(field->fwd_plan);
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
  if (debug)
    printf("Initializing field %s with N_i = %i, N_j = %i...\n",new_field->name,new_field->N_i, new_field->N_j);

  /* allocate and set up k */
  int i, j;
  new_field->kx = (double *) malloc(sizeof(double) * new_field->N_i);
  new_field->ky = (double *) malloc(sizeof(double) * new_field->N_j);
  for (i = 0; i < new_field->N_i; ++i)
    new_field->kx[i] = 2*M_PI*i;
  for (j = 0; j < new_field->N_j; ++j) {
    new_field->ky[j] = 2*M_PI*j;
  }
  /* Allocate memory for the real and k space data */
  new_field->kspace = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * new_field->N_i
                                                   * new_field->N_j);
  /* new_field->xspace = (double *) malloc(sizeof(double) * new_field->N_i */
  /*                                       * new_field->N_j); */
  new_field->xspace = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * new_field->N_i
                                                   * new_field->N_j);

  /* construct fftw plans */
  new_field->fwd_plan = fftw_plan_dft_2d(new_field->N_i, new_field->N_j, new_field->kspace, new_field->xspace, FFTW_FORWARD, FFTW_ESTIMATE);
  new_field->rev_plan = fftw_plan_dft_2d(new_field->N_i, new_field->N_j, new_field->xspace, new_field->kspace, FFTW_BACKWARD, FFTW_ESTIMATE);

  return;
}

void destroy_field(field *field) {
  free(field->kspace);
  free(field->xspace);
  fftw_destroy_plan(field->fwd_plan);
  fftw_destroy_plan(field->rev_plan);
}

int evolve_hydro_rk2(double dt, field *vx, field *vy) {
  /* take a Runge-Kutta-2 hydro timestep */

  /* 
     f_n+1/2 = (f_n + f_n+1)/2 = (f_n + f_n + h*RHS_n)/2 = f_n + h/2*RHS_n
     RHS_n+1/2 = RHS(f_n+1/2) = RHS(f_n) + h/2 * RHS(RHS_n)
     f_n+1 = f_n + h*RHS_n+1/2
   */

  /* from NR:
     k1 = h * RHS(x_n, y_n)
     k2 = h * RHS(x_n + 1/2*h, y_n + 1/2*k1)
     y_n+1 = y_n + k2 +O(h**3)
   */
  int i, j, N_j = vx->N_j;  
  field *RHS_x = create_field("RHS_x",vx->N_i, vx->N_j);
  field *RHS_y = create_field("RHS_y",vy->N_i, vy->N_j);
  field *vx1 = create_field("vx1",vx->N_i, vx->N_j);
  field *vy1 = create_field("vy1",vx->N_i, vx->N_j);
  init_field(RHS_x);
  init_field(RHS_y);
  init_field(vx1);
  init_field(vy1);
  /* first step */
  RHS(vx, vy, RHS_x, RHS_y);
  for (j = 0; j < vx->N_j; ++j)
    for (i = 0; i < vx->N_i; ++i) {
      vx1->kspace[index(i,j)][0] = vx->kspace[index(i,j)][0] + dt/2.*RHS_x->kspace[index(i,j)][0];
      vx1->kspace[index(i,j)][1] = vx->kspace[index(i,j)][1] + dt/2.*RHS_x->kspace[index(i,j)][1];
      vy1->kspace[index(i,j)][0] = vy->kspace[index(i,j)][0] + dt/2.*RHS_y->kspace[index(i,j)][0];
      vy1->kspace[index(i,j)][1] = vy->kspace[index(i,j)][1] + dt/2.*RHS_y->kspace[index(i,j)][1];
    }  

  RHS(vx1, vy1, RHS_x, RHS_y);
  for (j = 0; j < vx->N_j; ++j)
    for (i = 0; i < vx->N_i; ++i) {
      vx->kspace[index(i,j)][0] = vx->kspace[index(i,j)][0] + dt*RHS_x->kspace[index(i,j)][0];
      vx->kspace[index(i,j)][1] = vx->kspace[index(i,j)][1] + dt*RHS_x->kspace[index(i,j)][1];
      vy->kspace[index(i,j)][0] = vy->kspace[index(i,j)][0] + dt*RHS_y->kspace[index(i,j)][0];
      vy->kspace[index(i,j)][1] = vy->kspace[index(i,j)][1] + dt*RHS_y->kspace[index(i,j)][1];
    }
  destroy_field(vx1);
  destroy_field(vy1);
  destroy_field(RHS_x);
  destroy_field(RHS_y);
  return GOOD_STEP;
}

int RHS(field *vx, field *vy, field *RHS_x, field * RHS_y) {
  /* i hate C */

  int i, j, N_j = vx->N_j;  
  field *vgradvx = create_field("vgradvx",vx->N_i, vx->N_j);
  field *vgradvy = create_field("vgradvy",vy->N_i, vy->N_j);
  field *pressure = create_field("pressure",vy->N_i, vy->N_j);
  init_field(vgradvx);
  init_field(vgradvy);
  init_field(pressure);

  vgradv(vx, vy, vgradvx, vgradvy);
  calc_pressure(vgradvx, vgradvy, pressure);

  for (j = 0; j < pressure->N_j; ++j)
    for (i = 0; i < pressure->N_i; ++i) {
      RHS_x->kspace[index(i,j)][0] = pressure->kspace[index(i,j)][0]*pressure->kx[i] - vgradvx->kspace[index(i,j)][0];
      RHS_x->kspace[index(i,j)][1] = pressure->kspace[index(i,j)][1]*pressure->kx[i] - vgradvx->kspace[index(i,j)][1];
      RHS_y->kspace[index(i,j)][0] = pressure->kspace[index(i,j)][0]*pressure->ky[i] - vgradvy->kspace[index(i,j)][0];
      RHS_y->kspace[index(i,j)][1] = pressure->kspace[index(i,j)][1]*pressure->ky[i] - vgradvy->kspace[index(i,j)][1];
    }

  destroy_field(vgradvx);
  destroy_field(vgradvy);
  destroy_field(pressure);

  return GOOD_STEP;
}

int calc_pressure(field *vgradvx, field *vgradvy, field *pressure) {
  
  /* i/k FFT(d_j u_i  d_i u _j)

   */
  int i, j, N_j = pressure->N_j;  
  double ksquared;
  for (j = 0; j < pressure->N_j; ++j)
    for (i = 0; i < pressure->N_i; ++i) {
      ksquared = pressure->kx[i]*pressure->kx[i] 
        + pressure->ky[j]*pressure->ky[j];
      pressure->kspace[index(i,j)][0] = pressure->kx[i]*vgradvx->kspace[index(i,j)][1] + pressure->ky[i]*vgradvy->kspace[index(i,j)][1];
      pressure->kspace[index(i,j)][1] = pressure->kx[i]*vgradvx->kspace[index(i,j)][0] + pressure->ky[i]*vgradvy->kspace[index(i,j)][0];
    }
  return GOOD_STEP;
}

int vgradv(field *vx, field *vy, field *vgradvx, field *vgradvy) {
  /* compute non-linear term */
  /* WARNING: DEALIASING IS NOT YET IMPLEMENTED!!! SOLUTIONS ARE FULLY ALIASED! */

  int i, j, N_j = vx->N_j;
  field *dvxhatdx = create_field("dvxdx",vx->N_i, vx->N_j);
  field *dvyhatdx = create_field("dvydx",vy->N_i, vy->N_j);
  field *dvxhatdy = create_field("dvxdy",vx->N_i, vx->N_j);
  field *dvyhatdy = create_field("dvydy",vy->N_i, vy->N_j);
  init_field(dvxhatdx);
  init_field(dvyhatdx);
  init_field(dvxhatdy);
  init_field(dvyhatdy);

  dfhatdx(vx, dvxhatdx);
  dfhatdx(vy, dvyhatdx);
  dfhatdy(vx, dvxhatdy);
  dfhatdy(vy, dvyhatdy);

  /* compute vx, vy from kspace */
  fftw_execute(vx->fwd_plan);
  fftw_execute(vy->fwd_plan);

  /* compute dvx,dvy from kspace */
  fftw_execute(dvxhatdx->fwd_plan);
  fftw_execute(dvyhatdx->fwd_plan);
  fftw_execute(dvxhatdy->fwd_plan);
  fftw_execute(dvyhatdy->fwd_plan);

  /* compute v.grad(v) in real space */

  for (j = 0; j < vx->N_j; ++j) {
    for (i = 0; i < vx->N_i; ++i) {
      fflush(stdout);
      vgradvx->xspace[index(i,j)][0] = vx->xspace[index(i,j)][0] * dvxhatdx->xspace[index(i,j)][0] + vy->xspace[index(i,j)][0] * dvxhatdy->xspace[index(i,j)][0];
      vgradvy->xspace[index(i,j)][0] = vx->xspace[index(i,j)][0] * dvyhatdx->xspace[index(i,j)][0] + vy->xspace[index(i,j)][0] * dvyhatdy->xspace[index(i,j)][0];
    }
  }
  /* return to kspace */
  fftw_execute(vgradvx->rev_plan);
  fftw_execute(vgradvy->rev_plan);

  destroy_field(dvxhatdx);
  destroy_field(dvyhatdx);
  destroy_field(dvxhatdy);
  destroy_field(dvyhatdy);

  return GOOD_STEP;
}

int dfhatdx(field *f, field *dfhatdx) {
  /* compute kspace derivative */
  int i, j, N_j = f->N_j;

  for (j = 1; j < f->N_j; ++j)
    for (i = 1; i < f->N_j; ++i) {
      dfhatdx->kspace[index(i,j)][0] = -f->kspace[index(i,j)][1]*f->kx[i]; /* real */
      dfhatdx->kspace[index(i,j)][1] = f->kspace[index(i,j)][0]*f->kx[i];  /* imag */
    }

  return GOOD_STEP;
}

int dfhatdy(field *f, field *dfhatdy) {
  /* compute kspace derivative */
  int i, j, N_j = f->N_j;

  for (j = 1; j < f->N_j; ++j)
    for (i = 1; i < f->N_j; ++i) {
      dfhatdy->kspace[index(i,j)][0] = -f->kspace[index(i,j)][1]*f->ky[j]; /* real */
      dfhatdy->kspace[index(i,j)][1] = f->kspace[index(i,j)][0]*f->ky[j];  /* imag */
    }

  return GOOD_STEP;
}

int main() {
  int N_i = 100, N_j = 100;

  field *vx, *vy;
  vx = create_field("x-velocity",N_i,N_j);
  vy = create_field("y-velocity",N_i,N_j);

  init_field(vx);
  init_field(vy);
  tg_setup_2d(vx,vy);
  FILE *koutput;

  koutput = fopen("vx_tg_kspace_0.dat","w");

  write_field_kspace(koutput,vx);
  close(koutput);
  fftw_execute(vx->fwd_plan);

  /* main loop */
  double t = 0, dt=1e-4;
  double t_stop = 1;
  int it = 0, i_stop = 2;
  /* while (t < t_stop && it < i_stop) { */
  /*   printf("step %i\n", it); */
  /*   evolve_hydro_rk2(dt,vx,vy); */
    
  /*   t += dt; */
  /*   it++; */
  /* } */

  field *vxhat;
  vxhat = create_field("x-der",N_i,N_j);
  init_field(vxhat);
  dfhatdy(vx,vxhat);
  FILE *output;
  output = fopen("vx_tg_real.dat","w");
  koutput = fopen("vx_tg_kspace.dat","w");
  write_field_xspace(output,vxhat);
  write_field_kspace(koutput,vxhat);
  close(output);
  close(koutput);

  return 0;
}
