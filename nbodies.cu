#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <cub/cub.cuh>      //for parallel sum
#include <vector_types.h>   //for vector data types

#define n_bodies 1024
#define dt 1e-6
#define steps 2500000
#define L 100.0 // box size
#define G 1.0e3  //chose in a way such that it dominates over rectilinear uniform motion
#define eps 1e-1
#define save_step 100 //save output file every "save_step" steps
#define thr_per_b 32
#define n_blocks (n_bodies+thr_per_b-1)/thr_per_b
#define tile_dim thr_per_b
#define n_tiles n_blocks




__host__ float rand_interval(float min, float max){ //random float numbers in interval min,max
    return min + (float)rand()/RAND_MAX * (max-min);
}


__global__ void leapfrog_first_part(double3 *position_in,double3 *position_out, double3 *velocity, double3 *acc){

    int16_t gtid = blockIdx.x * blockDim.x + threadIdx.x; //global thread index

    if (gtid >= n_bodies)   //if n_bodies small
        return;

    velocity[gtid].x += 0.5 * acc[gtid].x * dt;
    velocity[gtid].y += 0.5 * acc[gtid].y * dt;
    velocity[gtid].z += 0.5 * acc[gtid].z * dt;

    position_out[gtid].x = position_in[gtid].x + velocity[gtid].x * dt;
    position_out[gtid].y = position_in[gtid].y + velocity[gtid].y * dt;
    position_out[gtid].z = position_in[gtid].z + velocity[gtid].z * dt;
}




__device__ void bodyBodyInteraction(double3 bi, double3 bj, float d_m, float m_i, double3 *ai, double *U, bool self_interaction){ //input: bi, bj, m; output: ai
    if(self_interaction)
        return; //do nothing

    double3 r; //declaring the vector r

    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    double r_2 = r.x * r.x + r.y * r.y + r.z * r.z + eps*eps;
    double r_6 = r_2 * r_2 * r_2;
    double r_m3 = 1.0/sqrt(r_6); //r^-3  m3: minus 3
    double s = G * d_m * r_m3;   //scalar factor; mass of j

    ai->x += r.x * s;   //ai->x instead of ai.x because ai is a pointer
    ai->y += r.y * s;
    ai->z += r.z * s;

    *U -= s*r_2*m_i*0.5; //dividing by 2 to avoid computing twice each pair i-j
}


__device__ void tile_calculation(double3 myPos, double3 *shPos, float *shMass, float m_i,double3 *acc, double *U, uint16_t gtid, uint16_t tile){ //myPos: position of the body for the executing thread

    for (uint16_t j=0; j < tile_dim; j++) {
        uint16_t idx = tile*tile_dim + j; //global intex of j
        bool self_interaction = (idx==gtid); //gloabal idx of j == global idx of i
        bodyBodyInteraction(myPos, shPos[j], shMass[j],m_i, acc,U,self_interaction);
    }
}


__global__ void calculate_forces(double3 *d_x, double3 *d_a, float *d_m, double *d_Etot){

    extern __shared__ unsigned char shmem[]; //extern: shared memory size defined at kernel launch
    double3* shPositions = (double3*) shmem; //pointer to the first part of shared memory
    float* shMasses = (float*)(shPositions + tile_dim);   

    uint16_t gtid = blockIdx.x * blockDim.x + threadIdx.x; //global thread index; each thread executing this command has a different scalar idx
    if (gtid >= n_bodies)   //if n_bodies small
        return;

    double3 myPos = d_x[gtid]; 

    double3 acc = {0.0, 0.0, 0.0};
    double U=0;
    float m_i=d_m[gtid];

    for (uint16_t tile = 0; tile < n_tiles; tile++)
    {
        uint16_t idx = tile * tile_dim + threadIdx.x;   //a scalar at each iteration; global index of the body to be loaded in shared memory

        if (idx < n_bodies) {
            shPositions[threadIdx.x] = d_x[idx];
            shMasses[threadIdx.x]    = d_m[idx];
        } else {                                               //if n_bodies small
            shPositions[threadIdx.x] = make_double3(0.0, 0.0, 0.0);
            shMasses[threadIdx.x]    = 0.0f;
        }
        __syncthreads(); //make sure all threads have loaded their data into shared memory
        tile_calculation(myPos, shPositions, shMasses,m_i, &acc, &U, gtid, tile);
        __syncthreads(); //to avoid overwriting shared memory
    }
    d_a[gtid] = acc;
    d_Etot[gtid] = U;
}




__global__ void leapfrog_final_part(double3 *d_v, double3 *d_a, double *d_Etot, float *d_m){

    uint16_t gtid = blockIdx.x * blockDim.x + threadIdx.x; 
    if (gtid >= n_bodies)   //if n_bodies small
        return;

    double K=0;

    d_v[gtid].x += 0.5 * d_a[gtid].x * dt;
    d_v[gtid].y += 0.5 * d_a[gtid].y * dt;
    d_v[gtid].z += 0.5 * d_a[gtid].z * dt;

    K=0.5*(d_v[gtid].x*d_v[gtid].x + d_v[gtid].y*d_v[gtid].y + d_v[gtid].z*d_v[gtid].z)*d_m[gtid];
    d_Etot[gtid] += K;
}












__host__ int main(){

    int saved_steps = (steps + save_step - 1) / save_step; //number of saved steps

    double3 *x[n_bodies];              //x is an array of n_bodies pointers to double3
    double3 *d_x_curr, *d_x_next, *d_v, *d_a;          //device pointers
    float *d_m;
    double *d_Etot;               //total energy, one for each body
    double *d_E_reduced;          //sum of energies
    void  *d_temp = NULL;         //for cub temporary storage; NULL means that we are asking cub to calculate the size we need
    size_t d_temp_size=0;         //for cub temporary storage size; it will be overwritten by cub after first call 

    for(uint16_t i=0;i<n_bodies;i++){                     //uint16_t: up to 65535 bodies
        x[i]= (double3*) malloc(saved_steps*sizeof(double3));   //memory allocated to each pointer
    }                                                     //each index contains a matrix steps x 3

    float *m = (float*)malloc(n_bodies*sizeof(float)); 
    double3 *v = (double3*)malloc(n_bodies*sizeof(double3));        //a matrix n_bodies x 3 
    double3 *a = (double3*)malloc(n_bodies*sizeof(double3)); 
    double *Etot =  (double*)malloc(steps * sizeof(double)); 

    //generating initial conditions on the host
    srand(31265);
    for(uint16_t i=0;i<n_bodies;i++){
        m[i]   = rand_interval(1.0f,10.0f);
        x[i][0].x = rand_interval(0.3f*L,0.7f*L);
        x[i][0].y = rand_interval(0.3f*L,0.7f*L);
        x[i][0].z = rand_interval(0.3f*L,0.7f*L);
        v[i].x = rand_interval(-1.0f,1.0f);
        v[i].y = rand_interval(-1.0f,1.0f);
        v[i].z = rand_interval(-1.0f,1.0f);
        a[i].x = a[i].y = a[i].z = 0.0f;
    }

    cudaEvent_t start_kernel,stop_kernel,start_alloc; 
    cudaEventCreate(&start_alloc);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);

    cudaEventRecord(start_alloc);

    //allocating and copying to the device
    cudaMalloc(&d_x_curr, n_bodies*sizeof(double3)); //the first argument is the address of the pointer where to allocate
    cudaMalloc(&d_x_next, n_bodies*sizeof(double3));
    cudaMalloc(&d_m, n_bodies*sizeof(float));
    cudaMalloc(&d_v,n_bodies*sizeof(double3));
    cudaMalloc(&d_a,n_bodies*sizeof(double3));
    cudaMalloc(&d_Etot, n_bodies*sizeof(double));   
    cudaMalloc(&d_E_reduced, save_step*sizeof(double));      
    
    cub::DeviceReduce::Sum(d_temp, d_temp_size, d_Etot, d_E_reduced, n_bodies); //first call to cub to get the size of d_temp_size; since d_temp is NULL, only d_temp_size is updated
    cudaMalloc(&d_temp, d_temp_size);                                              //allocating temporary storage for cub


    //cudaMemcpy(pointer to the destination, what to copy, size, type of transfer)
    for(uint16_t i = 0; i < n_bodies; i++){
        cudaMemcpy(&d_x_curr[i], &x[i][0], sizeof(double3), cudaMemcpyHostToDevice);  //copying only the initial positions
    }
    cudaMemcpy(d_m, m, n_bodies*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_v,v,n_bodies*sizeof(double3),cudaMemcpyHostToDevice);
    cudaMemcpy(d_a,a,n_bodies*sizeof(double3),cudaMemcpyHostToDevice);

    cudaDeviceSynchronize(); // Ensure all previous operations complete before timing
    cudaEventRecord(start_kernel);

    int save_idx=0;  
    cudaDeviceProp prop;     //for printing GPU properties          
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s, maxThreadsPerBlock=%d, sharedMemPerBlock=%zu\n",
           prop.name, prop.maxThreadsPerBlock, prop.sharedMemPerBlock);

    size_t shm_bytes = tile_dim * sizeof(double3) + tile_dim * sizeof(float);
    printf("launch config: blocks=%d, threads=%d, shm=%zu bytes, n_bodies=%d, steps=%d\n",
       n_blocks, thr_per_b, shm_bytes, n_bodies, steps);


    for(int step=0; step<steps; step++){
        leapfrog_first_part<<<n_blocks,thr_per_b>>>(d_x_curr,d_x_next,d_v, d_a);
        calculate_forces<<<n_blocks,thr_per_b, (tile_dim*sizeof(double3) + tile_dim*sizeof(float))>>>(d_x_next, d_a, d_m,d_Etot); //the third argument in <<<>>> is the size of dynamic shared memory
        leapfrog_final_part<<<n_blocks,thr_per_b>>>(d_v, d_a, d_Etot, d_m);

        //arguments: pointer to temporary storage, size of temporary storage, input data, output data, number of items
        cub::DeviceReduce::Sum(d_temp, d_temp_size, d_Etot, d_E_reduced + (step % save_step), n_bodies);

        cudaDeviceSynchronize(); //otherwise energy values can be modified while cub still working

                
        if(step % save_step ==0){  //copy only every save_step positions to the host
            cudaDeviceSynchronize(); //otherwise we can copy while the kernel is still running
            for(uint16_t i=0;i<n_bodies;i++){
                cudaMemcpy(&x[i][save_idx], &d_x_next[i], sizeof(double3), cudaMemcpyDeviceToHost);
            }
        }

        if((step+1) % save_step == 0){ //at step save_step - 1, 2*save_step - 1,...
            uint32_t host_offset = step - save_step + 1;  
            cudaMemcpy(&Etot[host_offset], d_E_reduced, save_step*sizeof(double), cudaMemcpyDeviceToHost);
            save_idx++;
        }

        double3 *tmp = d_x_curr;      //to swap pointers a tmp pointer is needed
        d_x_curr = d_x_next;
        d_x_next = tmp;
    }


    cudaDeviceSynchronize();
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);    

    // Print the time taken (in ms) between two events
    float elapsed_kernel, elapsed_alloc_to_end;
    
    cudaEventElapsedTime(&elapsed_kernel,start_kernel, stop_kernel);
    cudaEventElapsedTime(&elapsed_alloc_to_end,start_alloc, stop_kernel);

    printf("Elapsed time (kernel+copy):                 %.1f micros\n", elapsed_kernel*1000);
    printf("Elapsed time (allocation + kernel + copy): %.1f micros\n", elapsed_alloc_to_end*1000);

    // Destroy cudaEvents
    cudaEventDestroy(start_alloc);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    cudaFree(d_x_curr);cudaFree(d_x_next); cudaFree(d_v); cudaFree(d_Etot); cudaFree(d_m); cudaFree(d_a);
    cudaFree(d_E_reduced); cudaFree(d_temp);



    //output files for python usage; uncomment to use

/*
    FILE* fp = fopen("positions.csv", "w"); 	//fp: file positions

    fprintf(fp, "step,body,x,y,z\n");
    fprintf(fp,"\n");
  
    for (uint32_t t = 0; t < saved_steps; t++) {
        for (uint32_t i = 0; i < n_bodies; i++)
            fprintf(fp, "%d,%d,%.9f,%.9f,%.9f\n", t, i, x[i][t].x, x[i][t].y, x[i][t].z);//one line for each body; %d integer
    } 		 
    
    fclose(fp);  
*/


    FILE *fe = fopen("energy.csv","w");

    fprintf(fe,"Etot\n");
    for(uint32_t t=0;t<steps;t++){
        fprintf(fe,"%f\n",Etot[t]); 
    }

    fclose(fe); 


    //file xyz for ovito usage
/*    
    FILE* fxyz = fopen("trajectory.xyz", "w"); 	// fxyz: file XYZ

    for (uint32_t t = 0; t < saved_steps; t++) {
        fprintf(fxyz, "%d\n", n_bodies);    
        fprintf(fxyz, "Step: %d\n", t);    
        for (uint32_t i = 0; i < n_bodies; i++) {
            fprintf(fxyz, "%d %.9f %.9f %.9f\n", i, x[i][t].x, x[i][t].y, x[i][t].z);
        }
    } 		 

    fclose(fxyz);
*/    


    cudaFree(d_x_curr);cudaFree(d_x_next); cudaFree(d_v); cudaFree(d_Etot); cudaFree(d_m); cudaFree(d_a);
    cudaFree(d_E_reduced); cudaFree(d_temp);

    for(uint16_t i=0;i<n_bodies;i++){free(x[i]);}
    free(Etot); free(m); free(v); free(a);
    return 0;
}