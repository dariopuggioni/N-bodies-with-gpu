#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define n_bodies 1024
#define steps 150000
#define SRAND 31265

#ifdef FLOAT            //if FLOAT is defined during compilation with -DFLOAT
typedef float real;     //real is an alias of float
#define SQRT  sqrtf
#else
typedef double real;    
#define SQRT  sqrt
#endif


const float dt_max = 1e-6f;    //these variables can always be float
const float eta    = 1e-3f;
const float eps    = 1e-1f;
const float L      = 100.0f;
const float G      = 1.0e3f;

float rand_interval(float min, float max) {                   //for initializing float is enough
    return min + (float)rand()/RAND_MAX * (max - min);
}

void compute_initial_acceleration(const real* x, const real* y, const real* z,
                                  const float* m, real* ax, real* ay, real* az,
                                  real* U) {
    for (int i = 0; i < n_bodies; i++) ax[i] = ay[i] = az[i] = 0.0f;
    real U_local = 0.0f;

    for (int i = 0; i < n_bodies; i++) {
        for (int j = i + 1; j < n_bodies; j++) {
            real dx = x[j] - x[i];
            real dy = y[j] - y[i];
            real dz = z[j] - z[i];
            real r2 = dx*dx + dy*dy + dz*dz + eps*eps;
            real f = G / (r2 * SQRT(r2));

            ax[i] += f * m[j] * dx;
            ay[i] += f * m[j] * dy;
            az[i] += f * m[j] * dz;

            ax[j] -= f * m[i] * dx;
            ay[j] -= f * m[i] * dy;
            az[j] -= f * m[i] * dz;

            U_local -= G * m[i] * m[j] / SQRT(r2);
        }
    }
    *U = U_local;
}

int main(void) {
    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_MONOTONIC, &start);

    real* x  = malloc(n_bodies * sizeof(real));
    real* y  = malloc(n_bodies * sizeof(real));
    real* z  = malloc(n_bodies * sizeof(real));
    real* vx = malloc(n_bodies * sizeof(real));
    real* vy = malloc(n_bodies * sizeof(real));
    real* vz = malloc(n_bodies * sizeof(real));
    real* ax = malloc(n_bodies * sizeof(real));
    real* ay = malloc(n_bodies * sizeof(real));
    real* az = malloc(n_bodies * sizeof(real));
    float* m  = malloc(n_bodies * sizeof(float));

    real* x_new   = malloc(n_bodies * sizeof(real));
    real* y_new   = malloc(n_bodies * sizeof(real));
    real* z_new   = malloc(n_bodies * sizeof(real));
    real* vx_half = malloc(n_bodies * sizeof(real));
    real* vy_half = malloc(n_bodies * sizeof(real));
    real* vz_half = malloc(n_bodies * sizeof(real));
    real* ax_new  = malloc(n_bodies * sizeof(real));
    real* ay_new  = malloc(n_bodies * sizeof(real));
    real* az_new  = malloc(n_bodies * sizeof(real));

    real* Etot = malloc(steps * sizeof(real));
    float* time = malloc(steps * sizeof(float));

    #ifdef OUTPUT_POS                             //if OUTPUT_POS is defined during compilation with -DOUTPUT_POS
        FILE* fp = fopen("positions.csv", "w");  //fp: file positions
        fprintf(fp, "step,body,x,y,z\n");
    #endif

    srand(SRAND);
    for (int i = 0; i < n_bodies; i++) {
        m[i]  = rand_interval(1.0f, 10.0f);
        x[i]  = rand_interval(0.3f * L, 0.7f * L);
        y[i]  = rand_interval(0.3f * L, 0.7f * L);
        z[i]  = rand_interval(0.3f * L, 0.7f * L);
        vx[i] = rand_interval(-1.0f, 1.0f);
        vy[i] = rand_interval(-1.0f, 1.0f);
        vz[i] = rand_interval(-1.0f, 1.0f);
    }

    real U = 0.0f;
    compute_initial_acceleration(x, y, z, m, ax, ay, az, &U);

    real K = 0.0f;
    for (int i = 0; i < n_bodies; i++)
        K += 0.5f * m[i] * (vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);

    Etot[0] = K + U;
    time[0] = 0.0f;

    #ifdef OUTPUT_POS
        for (int i = 0; i < n_bodies; i++)
            fprintf(fp, "%d,%d,%.9f,%.9f,%.9f\n", 0, i, x[i], y[i], z[i]);
    #endif

    for (int step = 1; step < steps; step++) {
        real dt = dt_max;

        //while True
        while (1) {
            for (int i = 0; i < n_bodies; i++) {
                vx_half[i] = vx[i] + 0.5f * ax[i] * dt;
                vy_half[i] = vy[i] + 0.5f * ay[i] * dt;
                vz_half[i] = vz[i] + 0.5f * az[i] * dt;

                x_new[i] = x[i] + vx_half[i] * dt;
                y_new[i] = y[i] + vy_half[i] * dt;
                z_new[i] = z[i] + vz_half[i] * dt;
            }

            real min_r2 = 1e30f;
            int i_min = 0, j_min = 1;
            for (int i = 0; i < n_bodies; i++) {
                for (int j = i + 1; j < n_bodies; j++) {
                    real dx = x_new[j] - x_new[i];
                    real dy = y_new[j] - y_new[i];
                    real dz = z_new[j] - z_new[i];
                    real r2 = dx*dx + dy*dy + dz*dz + eps*eps;
                    if (r2 < min_r2) {min_r2 = r2; i_min = i; j_min = j;}
                }
            }

            real r_min = SQRT(min_r2);
            real dt_required = eta * SQRT((r_min*r_min*r_min) / (G * (m[i_min] + m[j_min])));
            if (dt <= dt_required) break;
            else {dt = dt_required;} //reduce dt and start again the loop
        }

        //after having checked dt
        for (int i = 0; i < n_bodies; i++) ax_new[i] = ay_new[i] = az_new[i] = 0.0f;
        U = 0.0f; K = 0.0f;

        for (int i = 0; i < n_bodies; i++) {
            for (int j = i + 1; j < n_bodies; j++) {
                real dx = x_new[j] - x_new[i];
                real dy = y_new[j] - y_new[i];
                real dz = z_new[j] - z_new[i];
                real r2 = dx*dx + dy*dy + dz*dz + eps*eps;
                real inv_r = 1.0f / SQRT(r2);
                real f = G * inv_r / r2;

                ax_new[i] += f * m[j] * dx;
                ay_new[i] += f * m[j] * dy;
                az_new[i] += f * m[j] * dz;

                ax_new[j] -= f * m[i] * dx;
                ay_new[j] -= f * m[i] * dy;
                az_new[j] -= f * m[i] * dz;

                U -= G * m[i] * m[j] * inv_r;
            }
        }

        for (int i = 0; i < n_bodies; i++) {
            vx[i] = vx_half[i] + 0.5f * ax_new[i] * dt;
            vy[i] = vy_half[i] + 0.5f * ay_new[i] * dt;
            vz[i] = vz_half[i] + 0.5f * az_new[i] * dt;

            x[i] = x_new[i];
            y[i] = y_new[i];
            z[i] = z_new[i];

            ax[i] = ax_new[i];
            ay[i] = ay_new[i];
            az[i] = az_new[i];

            K += 0.5f * m[i] * (vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
        }

        time[step] = time[step - 1] + dt;
        Etot[step] = K + U;

        #ifdef OUTPUT_POS
            if (step % 50 == 0) {
            for (int i = 0; i < n_bodies; i++)
                fprintf(fp, "%d,%d,%.9f,%.9f,%.9f\n", step, i, x[i], y[i], z[i]);  //one line for each body; %d integer
            }
        #endif
    }

    #ifdef OUTPUT_POS
        fclose(fp);
    #endif

    FILE* fe = fopen("energy.csv", "w");
    fprintf(fe, "time,Etot\n");
    for (int i = 0; i < steps; i++)
        fprintf(fe, "%.9f,%.9f\n", time[i], Etot[i]);
    fclose(fe);

    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("execution time: %.9f s\n", elapsed_time);

    free(x); free(y); free(z); free(m);
    free(vx); free(vy); free(vz);
    free(ax); free(ay); free(az);
    free(x_new); free(y_new); free(z_new);
    free(vx_half); free(vy_half); free(vz_half);
    free(ax_new); free(ay_new); free(az_new);
    free(Etot); free(time);

    return 0;
}