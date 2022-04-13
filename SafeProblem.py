import numpy as np
import array
import random
from deap import creator, base, tools, algorithms
from scipy.spatial import distance


def eval_on_grid(fun, xbound, n_steps):
    x_1 = np.linspace(xbound[0][0], xbound[0][1], n_steps)
    x_2 = np.linspace(xbound[1][0], xbound[1][1], n_steps)
    X,Y = np.meshgrid(x_1, x_2)
    # This is what safeopt.linearly_spaced_combinations() returns.
    x_matrix = np.vstack((X.ravel(), Y.ravel())).T
    x_matrix_2nd = np.vstack((Y.ravel(), X.ravel())).T
    y = np.empty(([n_steps*n_steps,1]))
    for i in range(n_steps*n_steps):
        y[i]=fun(x_matrix[i].tolist())
    return x_1, x_2, x_matrix,  x_matrix_2nd, y

def estimate_lipschitz(f, xbound, n_steps):
    x_1, x_2, x_matrix, x_matrix_2nd, y = eval_on_grid(f, xbound=xbound, n_steps = n_steps)
    
    GPforLipsX1 = np.empty(([n_steps*n_steps,1]))
    GPforLipsX2 = np.empty(([n_steps*n_steps,1]))
    for i in range(n_steps*n_steps):
        GPforLipsX1[i,0]=f(x_matrix[i].tolist())
        GPforLipsX2[i,0]=f(x_matrix_2nd[i].tolist())
    
    Gradient_info_x1_axis = np.zeros(n_steps)
    for each_axis1 in range(n_steps):
        Gradient_info_points = GPforLipsX1[each_axis1*n_steps:(each_axis1+1)*n_steps,0]
        Gradient_info_x1_axis[each_axis1] = np.max(abs(np.diff(Gradient_info_points)))
    
    Gradient_info_x2_axis = np.zeros(n_steps)
    for each_axis2 in range(n_steps):
        Gradient_info_points2 = GPforLipsX2[each_axis2*n_steps:(each_axis2+1)*n_steps,0]
        Gradient_info_x2_axis[each_axis2] = np.max(abs(np.diff(Gradient_info_points2)))

    return max([np.max(Gradient_info_x1_axis),np.max(Gradient_info_x2_axis)])*(n_steps-1)/(xbound[0][1] - xbound[0][0])

#bound
def checkBounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max:
                        child[i] = max
                    elif child[i] < min:
                        child[i] = min
            return offspring
        return wrapper
    return decorator

#out of bound is clipped to max or min


class Problem:
    def __init__(self, name, fun, bounds, percentile, default_safe_seeds,which_seed, problem):
        np.random.seed(which_seed+1)
        self._name = name
        self.fun = fun
        self.bounds = bounds
        self.xdim = len(bounds)
        assert percentile >= 0 and percentile < 0.96
        self.percentile = percentile
        n_steps = 500
        self.n_steps = n_steps
        self.x_1, self.x_2, self.x_matrix, self.x_matrix_2nd, self.y = eval_on_grid(self.fun, bounds, n_steps)
        self.safe_threshold = np.quantile(self.y, percentile)
        print(f'Safe Threshold ({self.percentile}) = {self.safe_threshold}')
        self._lipschitz = None # Lazy computation
        self.opt_x = None
        self.opt_y = None
        self.default_safe_seeds = default_safe_seeds
        self.problem = problem
        self._init_counters()

    @property
    def name(self): return self._name
    
    @property
    def lipschitz(self):
        # Lazy computation
        if self._lipschitz is None:
            self._lipschitz = estimate_lipschitz(f = self.fun, xbound = self.bounds, n_steps = 500)
            print(f'Lipschitz constant = {self.lipschitz}')
        return self._lipschitz
    
    def _init_counters(self):
        self.n_evaluations = 0
        self.n_unsafe = 0
        self.n_unsafe_evals = []
        self.Y = []
        self.Y_noise = []
        self.X1 = []
        self.X2 = []
        self.bsf = 0
        self.bsf_evals = []

    def is_safe(self, y):
        return y >= self.safe_threshold
        
    def _calculate_optimal(self):
        assert self.opt_x is None
        opt_pos = np.argmax(self.y)
        self.opt_y = self.y[opt_pos] 
        self.opt_x = self.x_matrix[opt_pos,:]
        assert self.opt_y == self.fun(self.opt_x)
        assert self.is_safe(self.opt_y)
        
    def get_optimal_x(self):
        if self.opt_x is None:
            self._calculate_optimal()
        return self.opt_x
    
    def get_optimal_y(self):
        if self.opt_y is None:
            self._calculate_optimal()
        return self.opt_y

    def get_default_safe_seeds(self, n=1):
        x_safe_seed = self.x_matrix[self.default_safe_seeds[:n],:]
        y_safe_seed = self.y[self.default_safe_seeds[:n]]
        y_safe_seed = y_safe_seed[:,None]
        print(f'Safe seeds:\n X = {x_safe_seed}\n y = {y_safe_seed}')
        assert np.all(self.is_safe(y_safe_seed))
        return x_safe_seed, y_safe_seed
        
    def get_uniform_safe_seeds(self, rng, n, beta):
        problem = self.problem
        if problem == 'sphere':
            q_val1 = self.safe_threshold + beta*0.1
            q_val2 = np.quantile(self.y, 0.96)
            while True:
                safe_region = (self.y > q_val1) & (self.y < q_val2)
                safe_idx = np.where(safe_region)[0]
                safe_idx = rng.choice(safe_idx, size = n, replace=False)
                x_safe_seed = self.x_matrix[safe_idx,:]
                y_safe_seed = self.y[safe_idx] + 0.1 * np.random.randn(n, 1)
                y_safe_seed_no_noise = self.y[safe_idx]
                if np.all(self.is_safe(y_safe_seed)):
                    print(f'Safe seeds:\n X = {x_safe_seed}\n y = {y_safe_seed}\n idx = {safe_idx}')
                    break
        
        elif problem == 'rastrigin':
            Divided_space = np.empty((self.y.shape[0],1))
            Divided_space[:,0] = 5
            for h in range(Divided_space.shape[0]):
                if (self.x_matrix[h][0]<=0) & (self.x_matrix[h][1]>=0):
                    Divided_space[h,0] = 0
                elif (self.x_matrix[h][0]>0) & (self.x_matrix[h][1]>=0):
                    Divided_space[h,0] = 1
                elif (self.x_matrix[h][0]<=0) & (self.x_matrix[h][1]<0):
                    Divided_space[h,0] = 2  
                elif (self.x_matrix[h][0]>0) & (self.x_matrix[h][1]<0):
                    Divided_space[h,0] = 3
            assert np.all(Divided_space!=5)
            q_val=np.quantile(self.y, 0.77)
            while True:
                safe_region1 = (self.y > (self.safe_threshold + beta*0.1)) & (self.y < q_val) & (Divided_space == 0)
                safe_region2 = (self.y > (self.safe_threshold + beta*0.1)) & (self.y < q_val) & (Divided_space == 1)
                safe_region3 = (self.y > (self.safe_threshold + beta*0.1)) & (self.y < q_val) & (Divided_space == 2)
                safe_region4 = (self.y > (self.safe_threshold + beta*0.1)) & (self.y < q_val) & (Divided_space == 3)
                safe_idx1 = np.where(safe_region1)[0]
                safe_idx2 = np.where(safe_region2)[0]
                safe_idx3 = np.where(safe_region3)[0]
                safe_idx4 = np.where(safe_region4)[0]
                safe_idx1 = rng.choice(safe_idx1, size = n//4, replace=False)
                safe_idx2 = rng.choice(safe_idx2, size = n//4, replace=False)
                safe_idx3 = rng.choice(safe_idx3, size = (n-3*n//4), replace=False)
                safe_idx4 = rng.choice(safe_idx4, size = n//4, replace=False)
                safe_idx = np.concatenate((safe_idx1,safe_idx2,safe_idx3,safe_idx4))
                x_safe_seed = self.x_matrix[safe_idx,:]
                y_safe_seed = self.y[safe_idx] + 0.1 * np.random.randn(n, 1)
                y_safe_seed_no_noise = self.y[safe_idx]
                if np.all(self.is_safe(y_safe_seed)):
                    print(f'Safe seeds:\n X = {x_safe_seed}\n y = {y_safe_seed}\n idx = {safe_idx}')
                    break
        elif problem == 'styblinski':
            q_val=np.quantile(self.y, 0.67)
            while True:
                safe_region = (self.y > (self.safe_threshold + beta*0.1)) & (self.y < q_val)
                safe_idx = np.where(safe_region)[0]
                safe_idx = rng.choice(safe_idx, size = n, replace=False)
                x_safe_seed = self.x_matrix[safe_idx,:]
                y_safe_seed = self.y[safe_idx] + 0.1 * np.random.randn(n, 1)
                y_safe_seed_no_noise = self.y[safe_idx]
                if np.all(self.is_safe(y_safe_seed)):
                    print(f'Safe seeds:\n X = {x_safe_seed}\n y = {y_safe_seed}\n idx = {safe_idx}')
                    break
        return x_safe_seed, y_safe_seed, y_safe_seed_no_noise
        
    def __call__(self, x):
        self.n_evaluations += 1
        y = self.fun(x)
        y_noise = self.fun(x) + 0.1 * np.random.randn(1, 1)[0][0]
        self.Y.append(y)
        self.Y_noise.append(y_noise)
        self.X1.append(x[0])
        self.X2.append(x[1])
        self.n_unsafe += int(~self.is_safe(y_noise))
        self.n_unsafe_evals = self.n_unsafe_evals + [self.n_unsafe]
        self.bsf = max(self.Y)
        self.bsf_evals = self.bsf_evals + [self.bsf]
        return y, y_noise

class EAProblemVA:
    def __init__(self, name, fun, fun_no_noise, MU, LAMBDA, bounds, percentile, default_safe_seeds, which_seed, problem):
        np.random.seed(which_seed+1)
        random.seed(which_seed+1)
        self._name = name
        self.fun = fun
        self.fun_no_noise = fun_no_noise
        self.bounds = bounds
        self.xdim = len(bounds)
        assert percentile >= 0 and percentile < 0.96
        self.percentile = percentile
        self.MU = MU
        self.LAMBDA = LAMBDA
        n_steps = 500
        self.n_steps = n_steps
        self.x_1, self.x_2, self.x_matrix, self.x_matrix_2nd, self.y = eval_on_grid(self.fun_no_noise, bounds, n_steps)
        self.safe_threshold = np.quantile(self.y, percentile)
        print(f'Safe Threshold ({self.percentile}) = {self.safe_threshold}')
        #self._lipschitz = None # Lazy computation
        self.opt_x = None
        self.opt_y = None
        self.default_safe_seeds = default_safe_seeds
        self.problem = problem
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()

        self.toolbox.register("Seed_Receive", random.uniform, self.bounds[0][0], self.bounds[0][1])
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.Seed_Receive, n=self.xdim)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.fun)
        self.toolbox.register("mate", tools.cxUniform, indpb = 0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu = 0, sigma = 0.1, indpb = 1)
        self.toolbox.register("select", tools.selTournament, tournsize=2)
        self.toolbox.decorate("mate", checkBounds(self.bounds[0][0], self.bounds[0][1]))
        self.toolbox.decorate("mutate", checkBounds(self.bounds[0][0], self.bounds[0][1]))
        self._init_counters()

    @property
    def name(self): return self._name
    
    #@property
    #def lipschitz(self):
        # Lazy computation
    #    if self._lipschitz is None:
    #        GPforLipsX1 = np.empty(([self.n_steps*self.n_steps,1]))
    #        GPforLipsX2 = np.empty(([self.n_steps*self.n_steps,1]))
    #        for i in range(self.n_steps*self.n_steps):
    #            GPforLipsX1[i,0]=self.fun(self.x_matrix[i].tolist())
    #            GPforLipsX2[i,0]=self.fun(self.x_matrix_2nd[i].tolist())

    #       GPSamGradX1 = np.empty(([self.n_steps*self.n_steps,1]))
    #        GPSamGradX2 = np.empty(([self.n_steps*self.n_steps,1]))

    #        GPSamGradX1[:,0] = abs(np.gradient(GPforLipsX1[:,0]))
    #        GPSamGradX2[:,0] = abs(np.gradient(GPforLipsX2[:,0]))

    #        self._lipschitz = max([np.max(GPSamGradX1,axis=0),np.max(GPSamGradX2,axis=0)])*(self.n_steps-1)/(self.bounds[0][1] - self.bounds[0][0])
    #        print(f'Lipschitz constant = {self.lipschitz}')
    #    return self._lipschitz

    def _init_counters(self):
        self.n_evaluations = 0
        self.n_generations = 0
        self.n_unsafe = 0
        self.n_unsafe_evals = []
        self.Y = []
        self.bsf = 0
        self.bsf_evals = []

    def is_safe(self, y):
        return y >= self.safe_threshold
        
    def _calculate_optimal(self):
        assert self.opt_x is None
        opt_pos = np.argmax(self.y)
        self.opt_y = self.y[opt_pos] 
        self.opt_x = self.x_matrix[opt_pos,:]
        assert self.opt_y == self.fun(self.opt_x)
        assert self.is_safe(self.opt_y)
        
    def get_optimal_x(self):
        if self.opt_x is None:
            self._calculate_optimal()
        return self.opt_x
    
    def get_optimal_y(self):
        if self.opt_y is None:
            self._calculate_optimal()
        return self.opt_y

    def get_default_safe_seeds(self, n=1):
        x_safe_seed = se
        lf.x_matrix[self.default_safe_seeds[:n],:]
        y_safe_seed = self.y[self.default_safe_seeds[:n]]
        y_safe_seed = y_safe_seed[:,None]
        print(f'Safe seeds:\n X = {x_safe_seed}\n y = {y_safe_seed}')
        assert np.all(self.is_safe(y_safe_seed))
        return x_safe_seed, y_safe_seed
        
    def get_uniform_safe_seeds(self,  n, beta):
        #random.seed(seed_number)
        #np.random.seed(seed_number)
        problem = self.problem
        population_candidates = self.toolbox.population(100000)
        invalid_ind = [ind for ind in population_candidates if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        population_for_sampling = list()
        if problem == 'sphere':
            q_val1 = self.safe_threshold + beta*0.1
            q_val2 = np.quantile(self.y, 0.96)
            for ind in population_candidates:
                if (ind.fitness.values[0] > q_val1) & (ind.fitness.values[0] < q_val2):
                    population_for_sampling.append(ind)
        if problem == 'rastrigin':
            q_val = np.quantile(self.y, 0.77)
            for ind in population_candidates:
                if (ind.fitness.values[0] > self.safe_threshold + beta*0.1) & (ind.fitness.values[0] < q_val):
                    population_for_sampling.append(ind)
        if problem == 'styblinski':
            q_val = np.quantile(self.y, 0.67)
            for ind in population_candidates:
                if (ind.fitness.values[0] > self.safe_threshold + beta*0.1) & (ind.fitness.values[0] < q_val):
                    population_for_sampling.append(ind)
        while True:
            random_samples = np.random.choice(len(population_for_sampling),size=n,replace=False)
            random_samples = list(random_samples)
            population = [population_for_sampling[i] for i in random_samples]

            safety_history = np.zeros(shape=(n,3))
            genome_location=[0,1]
            count=0
            for ind in population:
                safety_history[count,genome_location] = list(ind)
                count+=1
            initial_fitness_values = [None] * n
            for i in range(len(population)):
                initial_fitness_values[i] = population[i].fitness.values[0]
            y_safe_seed = np.array(initial_fitness_values)
            if np.all(self.is_safe(y_safe_seed)):
                break
        return population, safety_history 

 
    def __call__(self, population, offspring, safety_history):
        Mu_value= self.MU
        lambda_value = self.LAMBDA
        self.n_evaluations += lambda_value
        self.n_generations += 1
        self.current_Y = []
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        parents = offspring
        
        safety_entry = np.zeros(shape=(len(parents),3))
        count=0
        genome_location=[0,1]
        for ind in parents:
            safety_entry[count,genome_location] = list(ind)
            count+=1

        count=0
        for ind in parents:
            if ind.fitness.values[0] >= self.safe_threshold:
                safety_entry[count,2] = 0
            else:
                safety_entry[count,2] = 1
            count+=1
        safety_history_new = np.concatenate((safety_history,safety_entry),axis=0)
        
        population[:] = self.toolbox.select(population + offspring, Mu_value)
        y=np.empty((1,lambda_value))
        for k in range(lambda_value):
            y[0,k] = offspring[k].fitness.values[0]
            
        for h in range(lambda_value):
             self.current_Y.append(self.fun_no_noise(offspring[h])[0])
        self.Y = self.Y + self.current_Y
        
        self.n_unsafe += (lambda_value-np.sum(self.is_safe(y)))
        self.n_unsafe_evals = self.n_unsafe_evals + [self.n_unsafe]*len(self.current_Y)
        self.bsf = max(self.Y)
        self.bsf_evals = self.bsf_evals + [self.bsf]*len(self.current_Y)
        return population, safety_history_new

class EAProblem:
    def __init__(self, name, fun, fun_no_noise, MU, LAMBDA, bounds, percentile, default_safe_seeds, which_seed, problem):
        np.random.seed(which_seed+1)
        random.seed(which_seed+1)
        self._name = name
        self.fun = fun
        self.fun_no_noise = fun_no_noise
        self.bounds = bounds
        self.xdim = len(bounds)
        assert percentile >= 0 and percentile < 0.96
        self.percentile = percentile
        self.MU = MU
        self.LAMBDA = LAMBDA
        n_steps = 500
        self.n_steps = n_steps
        self.x_1, self.x_2, self.x_matrix, self.x_matrix_2nd, self.y = eval_on_grid(self.fun_no_noise, bounds, n_steps)
        self.safe_threshold = np.quantile(self.y, percentile)
        print(f'Safe Threshold ({self.percentile}) = {self.safe_threshold}')
        #self._lipschitz = None # Lazy computation
        self.opt_x = None
        self.opt_y = None
        self.default_safe_seeds = default_safe_seeds
        self.problem = problem
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()

        self.toolbox.register("Seed_Receive", random.uniform, self.bounds[0][0], self.bounds[0][1])
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.Seed_Receive, n=self.xdim)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.fun)
        self.toolbox.register("mate", tools.cxUniform, indpb = 0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu = 0, sigma = 0.1, indpb = 1)
        self.toolbox.register("select", tools.selTournament, tournsize=2)
        self.toolbox.decorate("mate", checkBounds(self.bounds[0][0], self.bounds[0][1]))
        self.toolbox.decorate("mutate", checkBounds(self.bounds[0][0], self.bounds[0][1]))
        self._init_counters()

    @property
    def name(self): return self._name
    
    #@property
    #def lipschitz(self):
        # Lazy computation
    #    if self._lipschitz is None:
    #        GPforLipsX1 = np.empty(([self.n_steps*self.n_steps,1]))
    #        GPforLipsX2 = np.empty(([self.n_steps*self.n_steps,1]))
    #        for i in range(self.n_steps*self.n_steps):
    #            GPforLipsX1[i,0]=self.fun(self.x_matrix[i].tolist())
    #            GPforLipsX2[i,0]=self.fun(self.x_matrix_2nd[i].tolist())

    #       GPSamGradX1 = np.empty(([self.n_steps*self.n_steps,1]))
    #        GPSamGradX2 = np.empty(([self.n_steps*self.n_steps,1]))

    #        GPSamGradX1[:,0] = abs(np.gradient(GPforLipsX1[:,0]))
    #        GPSamGradX2[:,0] = abs(np.gradient(GPforLipsX2[:,0]))

    #        self._lipschitz = max([np.max(GPSamGradX1,axis=0),np.max(GPSamGradX2,axis=0)])*(self.n_steps-1)/(self.bounds[0][1] - self.bounds[0][0])
    #        print(f'Lipschitz constant = {self.lipschitz}')
    #    return self._lipschitz

    def _init_counters(self):
        self.n_evaluations = 0
        self.n_generations = 0
        self.n_unsafe = 0
        self.n_unsafe_evals = []
        self.Y = []
        self.Y_noise = []
        self.bsf = 0
        self.bsf_evals = []

    def is_safe(self, y):
        return y >= self.safe_threshold
        
    def _calculate_optimal(self):
        assert self.opt_x is None
        opt_pos = np.argmax(self.y)
        self.opt_y = self.y[opt_pos] 
        self.opt_x = self.x_matrix[opt_pos,:]
        assert self.opt_y == self.fun(self.opt_x)
        assert self.is_safe(self.opt_y)
        
    def get_optimal_x(self):
        if self.opt_x is None:
            self._calculate_optimal()
        return self.opt_x
    
    def get_optimal_y(self):
        if self.opt_y is None:
            self._calculate_optimal()
        return self.opt_y

    def get_default_safe_seeds(self, n=1):
        x_safe_seed = se
        lf.x_matrix[self.default_safe_seeds[:n],:]
        y_safe_seed = self.y[self.default_safe_seeds[:n]]
        y_safe_seed = y_safe_seed[:,None]
        print(f'Safe seeds:\n X = {x_safe_seed}\n y = {y_safe_seed}')
        assert np.all(self.is_safe(y_safe_seed))
        return x_safe_seed, y_safe_seed
        
    def get_uniform_safe_seeds(self, n, beta):
        #random.seed(seed_number)
        #np.random.seed(seed_number)
        problem = self.problem
        population_candidates = self.toolbox.population(100000)
        invalid_ind = [ind for ind in population_candidates if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        population_for_sampling = list()
        if problem == 'sphere':
            q_val1 = self.safe_threshold + beta*0.1
            q_val2 = np.quantile(self.y, 0.96)
            for ind in population_candidates:
                if (ind.fitness.values[0] > q_val1) & (ind.fitness.values[0] < q_val2):
                    population_for_sampling.append(ind)
        if problem == 'rastrigin':
            q_val = np.quantile(self.y, 0.77)
            for ind in population_candidates:
                if (ind.fitness.values[0] > self.safe_threshold + beta*0.1) & (ind.fitness.values[0] < q_val):
                    population_for_sampling.append(ind)
        if problem == 'styblinski':
            q_val = np.quantile(self.y, 0.67)
            for ind in population_candidates:
                if (ind.fitness.values[0] > self.safe_threshold + beta*0.1) & (ind.fitness.values[0] < q_val):
                    population_for_sampling.append(ind)
        while True:
            random_samples = np.random.choice(len(population_for_sampling),size=n,replace=False)
            random_samples = list(random_samples)
            population = [population_for_sampling[i] for i in random_samples]

            initial_fitness_values = [None] * n
            for i in range(len(population)):
                initial_fitness_values[i] = population[i].fitness.values[0]
            y_safe_seed = np.array(initial_fitness_values)
            if np.all(self.is_safe(y_safe_seed)): 
                break
        return population

 
    def __call__(self, population, offspring):
        Mu_value= self.MU
        lambda_value = self.LAMBDA
        self.current_Y = []
        self.n_evaluations += lambda_value
        self.n_generations += 1
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        population[:] = self.toolbox.select(population + offspring, Mu_value)
        y=np.empty((1,lambda_value))
        y_noise = []
        for k in range(lambda_value):
            y[0,k] = offspring[k].fitness.values[0]
            y_noise = y_noise + [offspring[k].fitness.values[0]]
        for h in range(lambda_value):
             self.current_Y.append(self.fun_no_noise(offspring[h])[0])
        self.Y = self.Y + self.current_Y
        self.Y_noise = self.Y_noise + y_noise
        self.n_unsafe += (lambda_value-np.sum(self.is_safe(y)))
        self.n_unsafe_evals = self.n_unsafe_evals + [self.n_unsafe]*len(self.current_Y)
        self.bsf = max(self.Y)
        elf.bsf_evals = self.bsf_evals + [self.bsf]*len(self.current_Y)
        return population
