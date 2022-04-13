import GPy
from safeopt.gp_opt import SafeOpt
import numpy as np
import random
from numpy.random import default_rng
from deap import creator, base, tools, algorithms#
from scipy.spatial import distance

def run_modified_safeucb(fun, n_seeds, n_evals, which_seed):
    return run_safeopt(fun, n_seeds, n_evals, which_seed, modified = True, ucbselection = True)

def run_safeucb(fun, n_seeds, n_evals,  which_seed):
    return run_safeopt(fun, n_seeds, n_evals, which_seed, modified = False, ucbselection = True)

def run_modified_safeopt(fun, n_seeds, n_evals, which_seed):
    return run_safeopt(fun, n_seeds, n_evals, which_seed, modified = True)
    
def run_safeopt(fun, n_seeds, n_evals, which_seed, modified = False, ucbselection = False):
    random_seed = which_seed + 2
    rng = default_rng(random_seed)
    Y = []
    fun._init_counters()
    x_safe_seed, y_safe_seed, y_safe_seed_no_noise = fun.get_uniform_safe_seeds(rng, n = n_seeds, beta = 1.96)
    fun.y_initial_safe_seed = y_safe_seed
    fun.y_initial_safe_seed_no_noise = y_safe_seed_no_noise
    kernel = GPy.kern.RBF(input_dim=fun.xdim, ARD=False)
    #kernel = GPy.kern.Matern52(fun.xdim,ARD=True)
    # The statistical model of our objective function
    gp = GPy.models.GPRegression(x_safe_seed, y_safe_seed, kernel=kernel)
    gp.Gaussian_noise.variance = 0.01 
    gp.Gaussian_noise.variance.fix()
    if modified:
        lipschitz = None
    else:
        lipschitz = fun.lipschitz
        
    opt = SafeOpt(gp, parameter_set=fun.x_matrix, fmin=fun.safe_threshold,
                  lipschitz = lipschitz, beta = 2, # This was the default
                  threshold=0)

    assert n_evals > n_seeds
    for i in range(n_evals):
        # Obtain next query point
        if ucbselection:
            x_next = opt.optimize(ucb=False)
        else:
            x_next = opt.optimize(ucb=True)
        # Get a measurement from the real system
        y_no_noise, y_meas = fun(x_next)
        Y.append(y_no_noise)
        # Add this to the GP model
        opt.add_new_data_point(x_next, y_meas)
        print(f'evals={opt.t}\tx_next={x_next}\ty={y_meas}\ty_no_noise={y_no_noise}\tsafe={y_meas >= opt.fmin}\tbsf={max(Y)}')
    #return opt


def run_mu_plus_lambda_va(fun, n_seeds, n_evals):
    # n_evals = n_gens
    no_of_generations = (n_evals)//(n_seeds*2)
    #np.random.seed(1)
    #random.seed(1)
    fun._init_counters()
    population, safety_history_seed = fun.get_uniform_safe_seeds(n_seeds,  beta = 1.96)
    toolbox = fun.toolbox
    #X_to_be_returned = initial_population
    for gen in range(no_of_generations):
        # Vary the population
        offspring = algorithms.varOrVA(population, safety_history_seed, toolbox, lambda_=(n_seeds*2), cxpb=1/3, mutpb=1/3)
    
        # Evaluate the individuals with an invalid fitness
        population, safety_history_seed = fun(population, offspring, safety_history_seed)

        X_for_print = np.empty((len(offspring),2))
        Y_for_print = np.empty((len(offspring),1))
        Y_for_print_no_noise = np.empty((len(offspring),1))
        for i in range(len(offspring)):
            for j in range(2):
                X_for_print[i,j] = offspring[i][j]
        for i in range(len(offspring)):
                Y_for_print[i,0] = offspring[i].fitness.values[0]
        for i in range(len(offspring)):
                Y_for_print_no_noise[i,0] = fun.current_Y[i]
        ##bsf_individuals.append(current_bsf_individual)
        #X_to_be_returned = X_to_be_returned + population
        print(f'gens={gen+1}\tx_next={X_for_print}\ty={Y_for_print}\ty_no_noise={Y_for_print_no_noise}\tsafe={Y_for_print >= fun.safe_threshold}')
    #return population

def run_mu_plus_lambda(fun, n_seeds, n_evals):
    # n_evals = n_gens
    no_of_generations = (n_evals)//(n_seeds*2)
    #np.random.seed(1)
    #random.seed(1)
    fun._init_counters()
    population = fun.get_uniform_safe_seeds( n_seeds,  beta = 1.96)
    toolbox = fun.toolbox
    #X_to_be_returned = initial_population
    for gen in range(no_of_generations):
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_=(n_seeds*2), cxpb=1/3, mutpb=1/3)
    
        # Evaluate the individuals with an invalid fitness
        population = fun(population, offspring)

        X_for_print = np.empty((len(offspring),2))
        Y_for_print = np.empty((len(offspring),1))
        Y_for_print_no_noise = np.empty((len(offspring),1))
        for i in range(len(offspring)):
            for j in range(2):
                X_for_print[i,j] = offspring[i][j]
        for i in range(len(offspring)):
                Y_for_print[i,0] = offspring[i].fitness.values[0]
        for i in range(len(offspring)):
                Y_for_print_no_noise[i,0] = fun.current_Y[i]
        ##bsf_individuals.append(current_bsf_individual)
        #X_to_be_returned = X_to_be_returned + population
        print(f'gens={gen+1}\tx_next={X_for_print}\ty={Y_for_print}\ty_no_noise={Y_for_print_no_noise}\tsafe={Y_for_print >= fun.safe_threshold}')
    #return population
