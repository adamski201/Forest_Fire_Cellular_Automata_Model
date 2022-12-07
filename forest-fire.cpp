#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <vector>
#include <assert.h>
#include <omp.h>
#include <numeric>
#include <algorithm>

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
std::vector<std::vector<std::vector<double>>> forest_fire_average_steps(int lowerArraySize, int upperArraySize, int numberOfRuns);

int main(int argc, char **argv)
{
  // read the size of the (square) grid, the probability of filling a grid point with a tree and the random seed
  
  // check that we have 3 arguments (in addition to the program name)
  //   assert (argc == 4); 
  //   int N = atoi(argv[1]);
  //   double p = atof(argv[2]);
  //   int seed = atoi(argv[3]);
  
  int N = 100;
  double p = 0.6;
  int seed = 1; 
  /////////////////////////////////////////////////////////////////////
   
  // initialise the random number generator using a fixed seed for reproducibility
    srand(seed); 

    // call the forest fire function
    //int nsteps = forest_fire(N, p).stepCount;

    std::vector<std::vector<std::vector<double>>> result = forest_fire_average_steps(10, 110, 100);

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 21; j++)
        {
            for (int k = 0; k < result[i][j].size(); k++)
            {
                std::cout << result[i][j][k] << " ";
            }

            std::cout << std::endl;
        }

        std::cout << std::endl << std::endl;
    }
    
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

// Function that calculates the average number of steps, incrementing over arraysize and probability
// Output vector has 21 rows and columns represent [Probability, Average, Min Value, Max Value]
std::vector<std::vector<std::vector<double>>> forest_fire_average_steps(int lowerArraySize, int upperArraySize, int numberOfRuns)
{
    // Create empty vector to store results
    std::vector<std::vector<std::vector<double>>> results_vary_N_and_p;

    // Increment over arraysize
    for (int arraySize = lowerArraySize; arraySize <= upperArraySize; arraySize += 100)
    {
        std::vector<std::vector<double>> stepsResults;

        // init probability
        double p = 0;

        // Iterates over a range of probability values, from 0 to 1 in 0.05 increments.
        for (int i = 0; i < 21; ++i)
        {
            // Adds the probability value to the first column of the row.
            stepsResults.push_back(std::vector<double>());
            stepsResults[i].push_back(p);

            // Runs the forest fire model 'numberOfRuns' times with a defined array size
            // and stores the results.
            std::vector<double> runSteps;
            for (int a = 0; a < numberOfRuns; ++a)
            {
                runSteps.push_back(forest_fire(arraySize, p).stepCount);
            }

            // Calculates and stores the average of the results.
            double averageSteps = std::reduce(runSteps.begin(), runSteps.end()) / runSteps.size();
            stepsResults[i].push_back(averageSteps);
            stepsResults[i].push_back(*min_element(runSteps.begin(), runSteps.end()));
            stepsResults[i].push_back(*max_element(runSteps.begin(), runSteps.end()));

            // Increments probability.
            p += 0.05;
        }

        results_vary_N_and_p.push_back(stepsResults);
    }

    return results_vary_N_and_p;
}




