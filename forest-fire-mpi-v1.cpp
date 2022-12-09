#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <vector>
#include <assert.h>
#include <omp.h>
#include <numeric>
#include <algorithm>
#include <mpi.h>

/* 
- 0: no tree
- 1: tree is alive
- 2: tree is burning
- 3: tree is burnt */

// Define results struct
struct Results
{
    int stepCount;
    bool fireReachedEnd;
    double timeTaken;
};

// define arguments of the forest fire function
Results forest_fire(int N, double p);

int main(int argc, char **argv)
{
  int n_runs = 100; // Number of runs
  int seed = 1; 
  int arraySize = 100;
    /////////////////////////////////////////////////////////////////////
   
  // initialise the random number generator using a fixed seed for reproducibility
    srand(seed); 


    MPI_Init(nullptr, nullptr);

    int rank, n_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Determines which probabilities are handled by each process
    int start = rank * 21.0 / n_procs; 
    int end = std::min(((rank + 1) * 21.0 / n_procs), 21.0);

    // Initialise the probability step and results vectors.
    // We have 21 probabilities between 0 and 1 (inclusive).
    double prob_step = 0.05;
    std::vector<double> avg_steps_over_p(21,0);
    std::vector<int> min_steps_over_p(21,0);
    std::vector<int> max_steps_over_p(21,0);
    std::vector<double> prob_reached_end(21,0);

    // Loop over probabilities and compute the number of steps before the model burns out,
    // averaged over n_runs.
    for (int i = start; i < end; ++i)
    {
        double prob = i*prob_step;

        int n_steps = 0;
        int min_steps = std::numeric_limits<int>::max();
        int max_steps = 0;
        int n_reached_end = 0;

        for (int i_run = 0; i_run < n_runs; ++i_run)
        {
            Results result = forest_fire(arraySize, prob);
            
            n_steps += result.stepCount;

            if (result.fireReachedEnd) ++n_reached_end;
            if (result.stepCount < min_steps) min_steps = result.stepCount;
            if (result.stepCount > max_steps) max_steps = result.stepCount;
        }

        double p_reached_end = n_reached_end / n_runs;
        double avg_steps = n_steps / n_runs;
        avg_steps_over_p[i] = avg_steps;
        min_steps_over_p[i] = min_steps;
        max_steps_over_p[i] = max_steps;
        prob_reached_end[i] = p_reached_end;
    }

    // Worker processes communicate their results to the master process.
    if (rank > 0)
    {
        MPI_Send(&avg_steps_over_p[start], end-start, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
        MPI_Send(&min_steps_over_p[start], end-start, MPI_INT, 0, rank, MPI_COMM_WORLD);
        MPI_Send(&max_steps_over_p[start], end-start, MPI_INT, 0, rank, MPI_COMM_WORLD);
        MPI_Send(&prob_reached_end[start], end-start, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
    } else
    {
        for (int i = 1; i < n_procs; ++i)
        {
            int i_initial = 1.0*i / n_procs * 21.0;
            int i_final = std::min((1.0*(i+1)/ n_procs * 21.0),21.0);

            MPI_Status status;
            MPI_Recv(&avg_steps_over_p[i_initial], i_final-i_initial, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
            MPI_Recv(&min_steps_over_p[i_initial], i_final-i_initial, MPI_INT, i, i, MPI_COMM_WORLD, &status);
            MPI_Recv(&max_steps_over_p[i_initial], i_final-i_initial, MPI_INT, i, i, MPI_COMM_WORLD, &status);
            MPI_Recv(&prob_reached_end[i_initial], i_final-i_initial, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
        }

        // Master process outputs the final result.
        std::cout << "Probability, Avg. Steps, Min. Steps, Max Steps" << std::endl;
        for (int i = 0; i < 21; ++i)
        {
            double prob = i * prob_step;
            std::cout << prob << "," << avg_steps_over_p[i] 
                      << "," << min_steps_over_p[i] << "," 
                      << max_steps_over_p[i] << "," 
                      << prob_reached_end[i] << std::endl;
            
        }
    }

    MPI_Finalize();
    return 0;
}
  
// function which implements the forest fire model
Results forest_fire(int N, double p){

    // start timer
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);

  /* we want 2 grids - the currently active one, and the one from the previous time step -
     we call those grid_new and grid_old
     note: we could have generalised this to be a single 3 dimensional array/vector, with the 3rd index over time*/  
    std::vector < std::vector < int > > grid_old;
    // randomly fill the initial grid with live trees, using probability p
    for (unsigned int i=0;i<N;i++){
    grid_old.push_back(std::vector<int>());

        for (unsigned int j=0;j<N;j++){
            // generate a random floating point number between 0 and 1
            // note: random number generation is non-trivial - in real applications this approach may not be sufficiently random
            double rn = ((float)rand()/(float)(RAND_MAX));
        
            // if the random number is less than our probability p, we fill the site with a tree
            if (rn <= p){
                grid_old[i].push_back(1);
            }
            // otherwise, the site remains empty
            else{
                grid_old[i].push_back(0);
            }
        
        }
    }
  
    // set the top row of trees on fire
    for (unsigned int i=0;i<N;i++){
        if (grid_old[0][i] == 1) grid_old[0][i] = 2;
    }
  
    // initialise the new grid to an empty array
    std::vector < std::vector < int > > grid_new;
    for (unsigned int i=0;i<N;i++){
        grid_new.push_back(std::vector<int>());

        for (unsigned int j=0;j<N;j++){
            grid_new[i].push_back(0);
        }
    }
  
    // loop over time - this loop will continue until there are no more burning trees
    int t = 0;
    bool burning = true;
    bool fireReachedEnd = false;
    while (burning){

    // assume nothing is burning, unless we find otherwise below
    burning = false;

    // we want the following variables to be private:
    // i, j, left_burning, right_burning, above_burning and below_burning
    // and the following to be shared:
    // n, grid_old, grid_new, burning
    // by default, loop iteration variables and variables declared locally within the parallel loop are private
    // -> this covers all the private variables
    // variables declared outside the loop are shared
    // -> this covers all the shared variables
    // therefore we do not need to be explicit about any of the variables
    #pragma omp parallel for reduction (||:burning, fireReachedEnd)
    // loop over grid points
    for (unsigned int i=0;i<N;i++){
        for (unsigned int j=0;j<N;j++){
        
        // first check if this is a previously burning tree, which should change from burning (2) to burnt (3)
        // note: this operation depends only on the tree itself, i.e. the data at this grid point only
        if (grid_old[i][j] == 2){
            grid_new[i][j] = 3;
            burning = true;
        }
        
        // now we check whether this tree is alive
        else if (grid_old[i][j] == 1){
            /* in this case, we need to check the status of its neighbours, paying attention to whether or not we are at the edge
            note: this operation depends on data from other cells          
            we first set each variable to false, to cover the case where we are at the edge of the cell*/
            bool left_burning = false;
            if (j > 0) left_burning = (grid_old[i][j-1] == 2);
            
            bool right_burning = false;
            if (j < N-1) right_burning = (grid_old[i][j+1] == 2);
            
            bool below_burning = false;
            if (i < N-1) below_burning = (grid_old[i+1][j] == 2);
            
            bool above_burning = false;
            if (i > 0) above_burning = (grid_old[i-1][j] == 2);
            
            // if any of the neighbours are burning, set this tree on fire
            if (left_burning || right_burning || above_burning || below_burning){
            grid_new[i][j] = 2;
            }
            else{
            grid_new[i][j] = 1;
            }
        }
        
        // if there is no tree or it is already burnt, the status remains the same
        else{
            grid_new[i][j] = grid_old[i][j];
        }    
        }
    }
    // the OMP region ends here

    // increase the time counter
    t++;

    // the current grid will become the old grid in the next step
    grid_old = grid_new;

    // On the last step (once burning has stopped), checks if there are any burnt trees in the last row
    if (!burning)
    {
        for (int j = 0; j < N; ++j)
        {
            if (grid_new[N-1][j] == 3) fireReachedEnd = true;
        }
    }
    } 

    // end timer
    clock_gettime(CLOCK_MONOTONIC, &finish);
    double time_elapsed = (finish.tv_sec - start.tv_sec);
    time_elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    // store the results as a user-defined Results type
    Results result = {t, fireReachedEnd, time_elapsed};

    return result;
}