/* 
A simple 2-D Navier-Stokes solver for speedtesting pydro

Author: J. S. Oishi <jsoishi@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
License:
  Copyright (C) 2010-2011 J. S. Oishi.  All Rights Reserved.

  This file is part of pydro.

  pydro is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <fftw3.h>


/* this isn't really right. but it will work for now if you define a local variable called N_j... */
#define index(i,j) (j+N_j*i)
#define GOOD_STEP 0
#define BAD_STEP  1

int debug = 0;
double nu = 1e-2; /*1e-4; */

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

void write_field_xspace(field *field, int step) {
  char filen[50];
  sprintf(filen,"%s_xspace_%04i",field->name,step);
  FILE *file;
  file = fopen(filen,"w");
  fftw_execute(field->fwd_plan);
  int i,j, N_j;
  N_j = field->N_j;
  for (j = 0; j < field->N_j; j++) {
    for (i = 0; i < field->N_i; i++) { 
      fprintf(file, "%10.5e\n", field->xspace[index(i,j)][0]);
    }
  }
  close(file);
}

void write_field_kspace(field *field, int step) {
  char filen[50];
  sprintf(filen,"%s_kspace_%04i",field->name,step);
  FILE *file;
  file = fopen(filen,"w");
  int i,j, N_j;
  N_j = field->N_j;
  for (j = 0; j < N_j; j++) {
    for (i = 0; i < field->N_i; i++) { 
      fprintf(file, "%10.5e %10.5e\n", field->kspace[index(i,j)][0],field->kspace[index(i,j)][1]);
    }
  }
  close(file);
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
  free(field->kx);
  free(field->ky);
  fftw_destroy_plan(field->fwd_plan);
  fftw_destroy_plan(field->rev_plan);
  free(field);
}

int field_execute(field *field, int dir) {
  int i;
  if (dir == FFTW_FORWARD)
    fftw_execute(field->fwd_plan);
  if (dir == FFTW_BACKWARD) {
    fftw_execute(field->rev_plan);
    int N = field->N_i*field->N_j;
    for (i = 0; i < N; ++i) {
      field->kspace[i][0] /= (N*N);
      field->kspace[i][1] /= (N*N);
    }
  }
  
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
  double viscosity, k2;
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
      k2 = vx->kx[i] * vx->kx[i] + vx->ky[j] * vx->ky[j];
      viscosity = exp(-k2*dt/2.*nu);
      vx1->kspace[index(i,j)][0] = (vx->kspace[index(i,j)][0] + dt/2.*RHS_x->kspace[index(i,j)][0]) * viscosity;
      vx1->kspace[index(i,j)][1] = (vx->kspace[index(i,j)][1] + dt/2.*RHS_x->kspace[index(i,j)][1]) * viscosity;
      vy1->kspace[index(i,j)][0] = (vy->kspace[index(i,j)][0] + dt/2.*RHS_y->kspace[index(i,j)][0]) * viscosity;
      vy1->kspace[index(i,j)][1] = (vy->kspace[index(i,j)][1] + dt/2.*RHS_y->kspace[index(i,j)][1]) * viscosity;
    }  

  RHS(vx1, vy1, RHS_x, RHS_y);
  for (j = 0; j < vx->N_j; ++j)
    for (i = 0; i < vx->N_i; ++i) {
      k2 = vx->kx[i] * vx->kx[i] + vx->ky[j] * vx->ky[j];
      viscosity = exp(-k2*dt/2.*nu);
      vx->kspace[index(i,j)][0] = (vx->kspace[index(i,j)][0] * viscosity + dt*RHS_x->kspace[index(i,j)][0]) * viscosity;
      vx->kspace[index(i,j)][1] = (vx->kspace[index(i,j)][1] * viscosity + dt*RHS_x->kspace[index(i,j)][1]) * viscosity;
      vy->kspace[index(i,j)][0] = (vy->kspace[index(i,j)][0] * viscosity + dt*RHS_y->kspace[index(i,j)][0]) * viscosity;
      vy->kspace[index(i,j)][1] = (vy->kspace[index(i,j)][1] * viscosity + dt*RHS_y->kspace[index(i,j)][1]) * viscosity;
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
      RHS_x->kspace[index(i,j)][0] = -pressure->kspace[index(i,j)][1]*pressure->kx[i] - vgradvx->kspace[index(i,j)][0];
      RHS_x->kspace[index(i,j)][1] = -pressure->kspace[index(i,j)][0]*pressure->kx[i] - vgradvx->kspace[index(i,j)][1];
      RHS_y->kspace[index(i,j)][0] = -pressure->kspace[index(i,j)][1]*pressure->ky[i] - vgradvy->kspace[index(i,j)][0];
      RHS_y->kspace[index(i,j)][1] = -pressure->kspace[index(i,j)][0]*pressure->ky[i] - vgradvy->kspace[index(i,j)][1];
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
      if (ksquared == 0) {
          ksquared = 1;
        }
      pressure->kspace[index(i,j)][0] = (pressure->kx[i]*vgradvx->kspace[index(i,j)][1] + pressure->ky[i]*vgradvy->kspace[index(i,j)][1])/ksquared;
      pressure->kspace[index(i,j)][1] = (pressure->kx[i]*vgradvx->kspace[index(i,j)][0] + pressure->ky[i]*vgradvy->kspace[index(i,j)][0])/ksquared;
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
  field_execute(vx, FFTW_FORWARD);
  field_execute(vy, FFTW_FORWARD);

  /* compute dvx,dvy from kspace */
  field_execute(dvxhatdx, FFTW_FORWARD);
  field_execute(dvyhatdx, FFTW_FORWARD);
  field_execute(dvxhatdy, FFTW_FORWARD);
  field_execute(dvyhatdy, FFTW_FORWARD);

  /* compute v.grad(v) in real space */

  for (j = 0; j < vx->N_j; ++j) {
    for (i = 0; i < vx->N_i; ++i) {
      /* fflush(stdout); */
      vgradvx->xspace[index(i,j)][0] = vx->xspace[index(i,j)][0] * dvxhatdx->xspace[index(i,j)][0] + vy->xspace[index(i,j)][0] * dvxhatdy->xspace[index(i,j)][0];
      vgradvx->xspace[index(i,j)][1] = 0;
      vgradvy->xspace[index(i,j)][0] = vx->xspace[index(i,j)][0] * dvyhatdx->xspace[index(i,j)][0] + vy->xspace[index(i,j)][0] * dvyhatdy->xspace[index(i,j)][0];
      vgradvy->xspace[index(i,j)][1] = 0;
    }
  }
  /* return to kspace */
  field_execute(vgradvx,FFTW_BACKWARD);
  field_execute(vgradvy,FFTW_BACKWARD);

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

int main(int argc, char *argv[]) {
  int N_i = 100, N_j = 100;

  field *vx, *vy;
  vx = create_field("x-velocity",N_i,N_j);
  vy = create_field("y-velocity",N_i,N_j);

  init_field(vx);
  init_field(vy);
  tg_setup_2d(vx,vy);

  write_field_kspace(vx,0);
  write_field_xspace(vx,0);
  write_field_kspace(vy,0);
  write_field_xspace(vy,0);

  field_execute(vx, FFTW_FORWARD);

  /* main loop */
  double t = 0, dt=1e-6;
  double t_stop = 1;
  int it = 0, i_stop = 10;
  if (argc == 2) 
    i_stop = atoi(argv[1]);
  
  printf("running %i steps\n", i_stop);
  while (t < t_stop && it < i_stop) {
    printf("step %i\n", it);
    evolve_hydro_rk2(dt,vx,vy);
    
    t += dt;
    it++;
    write_field_xspace(vx,it);
    write_field_xspace(vy,it);
  }

  write_field_kspace(vx,it);
  write_field_kspace(vy,it);
  destroy_field(vx);
  destroy_field(vy);
  return 0;
}
