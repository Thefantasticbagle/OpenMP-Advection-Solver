/**
 * An OpenMP solver for the Advection Problem.
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

#include <omp.h>

#include "../inc/utils.h"

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

typedef int64_t int_t;
typedef double real_t;

int_t
    thread_count = 10,
    y_size,
    x_size,
    iterations,
    snapshot_frequency;

real_t
    *temp[2] = { NULL, NULL },
    *thermal_diffusivity,
    dt;

#define T(x,y)                      temp[0][(y) * (x_size + 2) + (x)]
#define T_next(x,y)                 temp[1][((y) * (x_size + 2) + (x))]
#define THERMAL_DIFFUSIVITY(x,y)    thermal_diffusivity[(y) * (x_size + 2) + (x)]

void time_step ( void );
void boundary_condition( void );
void border_exchange( void );
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

    // Initialize
    domain_init();

    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );

    // Iterate...
    for ( int_t iteration = 0; iteration <= iterations; iteration++ )
    {
        // Handle boundary conditions
        boundary_condition();

        // Make one time-step
        time_step();

        // Take snapshot
        if ( iteration % snapshot_frequency == 0 )
        {
            printf (
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                iterations,
                100.0 * (real_t) iteration / (real_t) iterations
            );

            domain_save ( iteration );
        }

        // Swap prev and next to prepare next iteration
        swap( &temp[0], &temp[1] );
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
 * Calculates one time-step for the entire grid.
*/
void
time_step ( void )
{
#   pragma omp parallel for num_threads( thread_count ) collapse(2)
    for ( int_t y = 1; y <= y_size; y++ )
    {
        for ( int_t x = 1; x <= x_size; x++ )
        {
            real_t c, t, b, l, r, K, new_value;
            c = T(x, y);

            t = T(x - 1, y);
            b = T(x + 1, y);
            l = T(x, y - 1);
            r = T(x, y + 1);
            K = THERMAL_DIFFUSIVITY(x, y);

            new_value = c + K * dt * ((l - 2 * c + r) + (b - 2 * c + t));

            T_next(x, y) = new_value;
        }
    }
}


/**
 * Handles boundary conditions.
*/
void
boundary_condition ( void )
{
#   pragma omp parallel for num_threads( thread_count )
    for ( int_t x = 1; x <= x_size; x++ )
    {
        T(x, 0) = T(x, 2);
        T(x, y_size+1) = T(x, y_size-1);
    }

#   pragma omp parallel for num_threads( thread_count )
    for ( int_t y = 1; y <= y_size; y++ )
    {
        T(0, y) = T(2, y);
        T(x_size+1, y) = T(x_size-1, y);
    }
}


/**
 * Initializes the grid.
 * Remember to finalize the domain when its variables are no longer in use!
 * 
 * @see domain_finalize()
*/
void
domain_init ( void )
{
    temp[0] = malloc ( (y_size+2)*(x_size+2) * sizeof(real_t) );
    temp[1] = malloc ( (y_size+2)*(x_size+2) * sizeof(real_t) );
    thermal_diffusivity = malloc ( (y_size+2)*(x_size+2) * sizeof(real_t) );

    dt = 0.1;

#   pragma omp parallel for num_threads( thread_count ) collapse(2)
    for ( int_t y = 1; y <= y_size; y++ )
    {
        for ( int_t x = 1; x <= x_size; x++ )
        {
            real_t temperature = 30 + 30 * sin((x + y) / 20.0);
            real_t diffusivity = 0.05 + (30 + 30 * sin((x_size - x + y) / 20.0)) / 605.0;

            T(x,y) = temperature;
            T_next(x,y) = temperature;
            THERMAL_DIFFUSIVITY(x,y) = diffusivity;
        }
    }
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

    fwrite( temp[0], sizeof(real_t), (x_size+2)*(y_size+2), out );
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
    free ( temp[0] );
    free ( temp[1] );
    free ( thermal_diffusivity );
}
