/**
 * A CUDA solver for the Advection Problem.
 * https://en.wikipedia.org/wiki/Advection
 * 
 * @file advection_solver.cu
 * @author Lars L Ruud
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>

#include "../inc/utils.h"

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

typedef int64_t int_t;
typedef double real_t;

int_t
    y_size,
    x_size,
    iterations,
    snapshot_frequency,
    blocks,
    threads_per_block;

real_t
    *h_temp[2] = { NULL, NULL },
    *h_thermal_diffusivity,
    *d_temp,
    *d_temp_next,
    *d_thermal_diffusivity,
    dt;

#define T(x,y)                      h_temp[0][(y) * (x_size + 2) + (x)]
#define T_next(x,y)                 h_temp[1][((y) * (x_size + 2) + (x))]
#define THERMAL_DIFFUSIVITY(x,y)    h_thermal_diffusivity[(y) * (x_size + 2) + (x)]

#define d_T(x,y)                      d_temp[(y) * (x_size + 2) + (x)]
#define d_T_next(x,y)                 d_temp_next[((y) * (x_size + 2) + (x))]
#define d_THERMAL_DIFFUSIVITY(x,y)    d_thermal_diffusivity[(y) * (x_size + 2) + (x)]

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void time_step ( real_t *d_temp, real_t *d_temp_next, real_t *d_thermal_diffusivity, int_t x_size, int_t y_size, real_t dt);
__device__ void boundary_condition( real_t *d_temp, int_t x, int_t y, int_t x_size, int_t y_size );
void domain_init ( void );
void domain_save ( int_t iteration );
void domain_finalize ( void );


/**
 * Swaps the values of two arrays.
 * 
 * @param m1 The first array.
 * @param m2 The second array.
*/
void
swap ( real_t** m1, real_t** m2 )
{
    real_t* tmp;
    tmp = *m1;
    *m1 = *m2;
    *m2 = tmp;
}


/**
 * The main function.
*/
int
main ( int argc, char **argv )
{
    // Parse arguments
    ARGS *args = parse_args( argc, argv );
    if ( !args )
    {
        fprintf( stderr, "Argument parsing failed\n" );
        exit(1);
    }

    y_size = args->y_size;
    x_size = args->x_size;
    iterations = args->iterations;
    snapshot_frequency = args->snapshot_frequency;

    // Each pixel of the grid is assigned a thread.
    //  In the future I might add some optimization so that it automatically
    //  matches the SM (blocks) and SP (threads per block) of the GPU of the user.
    blocks = y_size;
    threads_per_block = x_size;

    // Initialize
    domain_init();

    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );

    // Iterate...
    for ( int_t iteration = 0; iteration <= iterations; iteration++ )
    {
        // Launch kernels
        time_step <<<blocks, threads_per_block>>>( d_temp, d_temp_next, d_thermal_diffusivity, x_size, y_size, dt );

        // Take snapshot
        if ( iteration % snapshot_frequency == 0 )
        {
            printf (
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                iterations,
                100.0 * (real_t) iteration / (real_t) iterations
            );

            // Copy data to host and save
            cudaMemcpy ( h_temp[0], d_temp, (y_size+2)*(x_size+2) * sizeof(real_t), cudaMemcpyDeviceToHost );
            domain_save ( iteration );
        }

        // Swap prev and next to prepare next iteration
        swap ( &d_temp, &d_temp_next );
    }

    // Take the time
    gettimeofday ( &t_end, NULL );
    printf ( "Total elapsed time: %lf seconds\n",
            WALLTIME(t_end) - WALLTIME(t_start)
            );

    // Clean up and exit
    domain_finalize();

    exit ( EXIT_SUCCESS );
}


/**
 * Calculates one time-step for one pixel.
 * 
 * @param d_temp A pointer to temp-array in the device's memory.
 * @param d_temp_next A pointer to the temp_next-array in the device's memory.
 * @param d_thermal_diffusivity A pointer to the d_thermal_diffusivity-array in the device's memory.
 * @param x_size How many pixels wide the grid is.
 * @param y_size How many pixels tall the grid is.
 * @param dt How large of a time-step to take. 
*/
__global__ void
time_step ( real_t *d_temp, real_t *d_temp_next, real_t *d_thermal_diffusivity, int_t x_size, int_t y_size, real_t dt )
{
    // Get which gridpoint the thread is responsible for
    int_t   rank = blockDim.x * blockIdx.x + threadIdx.x,
            x = rank % x_size + 1,
            y = rank / x_size + 1;

    // (Skip if it lies outside of the grid, although this should never happen since threads = y_size * x_size)
    if (y > y_size) return;

    // Do boundary condition
    boundary_condition ( d_temp, x, y, x_size, y_size );

    // Do time-step computation
    real_t c, t, b, l, r, K, new_value;

    c = d_T(x, y);
    t = d_T(x - 1, y);
    b = d_T(x + 1, y);
    l = d_T(x, y - 1);
    r = d_T(x, y + 1);
    K = d_THERMAL_DIFFUSIVITY(x, y);

    new_value = c + K * dt * ((l - 2 * c + r) + (b - 2 * c + t));
    d_T_next(x, y) = new_value;
}


/**
 * Handles boundary conditions.
 * 
 * @param A pointer to temp-array in the device's memory.
 * @param x The x coordinate of the current pixel in the grid.
 * @param y The y coordinate of the current pixel in the grid.
 * @param x_size How many pixels wide the grid is.
 * @param y_size How many pixels tall the grid is.
 * 
 * @see time_step(...)
*/
__device__ void
boundary_condition ( real_t *d_temp, int_t x, int_t y, int_t x_size, int_t y_size )
{ 
    if ( y == 1 ) {
        d_T(x, 0) = d_T(x, 2);
    } else if ( y == y_size ) {
        d_T(x, y_size+1) = d_T(x, y_size-1);
    }

    if ( x == 1 ) {
        d_T(0, y) = d_T(2, y);
    } else if (x == x_size) {
        d_T(x_size+1, y) = d_T(x_size-1, y);
    }
}


/**
 * Initializes the grid, both host-side and device-side.
 * Remember to finalize the domain when its variables are no longer in use!
 * 
 * @see domain_finalize()
*/
void
domain_init ( void )
{
    // Allocate host-side memory
    size_t arena_size = (y_size+2)*(x_size+2) * sizeof(real_t);
    h_temp[0] = (real_t*) malloc ( arena_size );
    h_temp[1] = (real_t*) malloc ( arena_size );
    h_thermal_diffusivity = (real_t*) malloc ( arena_size );

    // Allocate device-side memory
    cudaMalloc ( &d_temp, arena_size );
    cudaMalloc ( &d_temp_next, arena_size );
    cudaMalloc ( &d_thermal_diffusivity, arena_size );

    // Set starting conditions
    dt = 0.1;

    for ( int_t y = 1; y <= y_size; y++ )
    {
        for ( int_t x = 1; x <= x_size; x++ )
        {
            real_t temperature = 30 + 30 * sin((x + y) / 20.0);
            real_t diffusivity = 0.05 + (30 + 30 * sin((x_size - x + y) / 20.0)) / 605.0;

            h_temp[0][ y*(x_size+2) + x ] = temperature;
            h_temp[1][ y*(x_size+2) + x ] = temperature;
            h_thermal_diffusivity[ y*(x_size+2) + x ] = diffusivity;
        }
    }

    // Copy host-memory onto device
    cudaMemcpy ( d_temp, h_temp[0], arena_size, cudaMemcpyHostToDevice );
    cudaMemcpy ( d_temp_next, h_temp[1], arena_size, cudaMemcpyHostToDevice );
    cudaMemcpy ( d_thermal_diffusivity, h_thermal_diffusivity, arena_size, cudaMemcpyHostToDevice );
}


/**
 * Saves the grid.
 * Writes to the data/ folder.
 * 
 * @param iteration Which iteration the current state of the grid is for.
*/
void
domain_save ( int_t iteration )
{
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset ( filename, 0, 256*sizeof(char) );
    sprintf ( filename, "data/%.5ld.bin", index );

    FILE *out = fopen ( filename, "wb" );
    if ( ! out ) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(1);
    }
    for ( int_t iter = 1; iter <= x_size; iter++)
    {
        fwrite( h_temp[0] + (y_size+2) * iter + 1, sizeof(real_t), x_size, out );
    }
    fclose ( out );
}


/**
 * De-allocates the grid from memory.
 * 
 * @see domain_initialize()
*/
void
domain_finalize ( void )
{
    // Free host-memory
    free ( h_temp[0] );
    free ( h_temp[1] );
    free ( h_thermal_diffusivity );

    // Free device-memory
    cudaFree ( d_temp );
    cudaFree ( d_temp_next );
    cudaFree ( d_thermal_diffusivity );
}
