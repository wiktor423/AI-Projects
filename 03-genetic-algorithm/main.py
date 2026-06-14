import numpy as np
import matplotlib.pyplot as plt 
import random as rand


def min_max_norm(val, min_val, max_val, new_min, new_max):
    return (val - min_val) * (new_max - new_min) / (max_val - min_val) + new_min

def rand_vec(length):
    vec=""
    for i in range (length): 
        temp = str(rand.randInt(0,1))
        vec+=temp
    return vec

class Chromosome:
    def __init__(self, length, array=None):  # if array is None, initialize with a random binary vector
        self.length = length
        if array is None:
            self.array = np.random.randint(0,2, size=length)
        else: 
            self.array = np.array(array)

    def decode(self, lower_bound, upper_bound, aoi):
        segment = self.array[lower_bound:upper_bound]
        bit_string = "".join(segment.astype(str)) 
        value = int(bit_string, 2) 

        segment_length = upper_bound - lower_bound
        max_value = 2 ** segment_length - 1
        decoded = min_max_norm(value, 0, max_value, aoi[0], aoi[1])
        return decoded


    def mutation(self, probability):    
        p = rand.random() 
        if p < probability: 
    
            idx = rand.randrange(0, self.length)
            self.array[idx] = 1 - self.array[idx]
        
    def crossover(self, other):
        crossover_point = rand.randrange(1, self.length - 1)
        child1_array = np.concatenate((self.array[:crossover_point], other.array[crossover_point:]))
        child2_array = np.concatenate((other.array[:crossover_point], self.array[crossover_point:]))
        return Chromosome(self.length, child1_array), Chromosome(self.length, child2_array)
        pass    

        

# TODO: implement your group's objective function here
def objective_function(*args):
    fitness = (args[0]**2 + args[1] - 11)**2 + (args[0] + args[1]**2 - 7)**2
    return fitness


class GeneticAlgorithm:
    def __init__(self, chromosome_length, obj_func_num_args, objective_function, aoi,
                 population_size=100, tournament_size=2, mutation_probability=0.05,
                 crossover_probability=0.8, num_steps=50):
        assert chromosome_length % obj_func_num_args == 0, "Number of bits for each argument should be equal"

        self.chromosome_length = chromosome_length
        self.obj_func_num_args = obj_func_num_args
        self.bits_per_arg = int(chromosome_length / obj_func_num_args)
        self.objective_function = objective_function
        self.aoi = aoi
        self.tournament_size = tournament_size
        self.mutation_probability = mutation_probability
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.num_steps = num_steps

    def eval_objective_func(self, chromosome):
        args = []

        for i in range(self.obj_func_num_args):
            lower_bound = i * self.bits_per_arg
            upper_bound = (i + 1) * self.bits_per_arg
            arg = chromosome.decode(lower_bound, upper_bound, self.aoi)
            args.append(arg)
        return self.objective_function(*args)  

    def tournament_selection(self):
        tournament = rand.sample(self.population, self.tournament_size)
        for chromo in tournament:
            chromo.fitness = self.eval_objective_func(chromo)
        best = min(tournament, key=lambda x: x.fitness)
        return best  

    def reproduce(self, parents):
        if rand.random() < self.crossover_probability: 
            children = parents[0].crossover(parents[1])
            children = list(children) 
        else: 
            child1 = Chromosome(self.chromosome_length, parents[0].array.copy())
            child2 = Chromosome(self.chromosome_length, parents[1].array.copy())
            children = [child1, child2]

        children[0].mutation(self.mutation_probability)
        children[1].mutation(self.mutation_probability)
        return children 

    def plot_func(self, trace, title="Himmelblau - GA trace"):
       
        x1 = np.linspace(self.aoi[0], self.aoi[1], 300)
        x2 = np.linspace(self.aoi[0], self.aoi[1], 300)
        X1, X2 = np.meshgrid(x1, x2)
        Z = self.objective_function(X1, X2)

        plt.figure(figsize=(8, 6))
        plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
        plt.colorbar()

        colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(trace)))
        for i, (args, fitness) in enumerate(trace):
            plt.scatter(args[0], args[1], color=colors[i], s=20, zorder=5)

        plt.title(title)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()
        
        
    def run(self):
        self.population = [Chromosome(self.chromosome_length) for _ in range(self.population_size)]
        
        trace = []

        for step in range(self.num_steps):
            new_population = []
            while len(new_population) < self.population_size:

                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                parents = [parent1, parent2]

                children = self.reproduce(parents)
                new_population.extend(children)

            self.population = new_population

            for chromosome in self.population:
                chromosome.fitness = self.eval_objective_func(chromosome)
            best_chromosome = min(self.population, key=lambda x: x.fitness)

            best_args = []
            for i in range(self.obj_func_num_args):
                lower_bound = i * self.bits_per_arg
                upper_bound = (i + 1) * self.bits_per_arg
                arg = best_chromosome.decode(lower_bound, upper_bound, self.aoi)
                best_args.append(arg)
                
            trace.append((best_args, best_chromosome.fitness))

        return trace




# TODO: fill in the parameters for your group and uncomment to run
ga = GeneticAlgorithm(
    chromosome_length=16,
    obj_func_num_args=2,
    objective_function=objective_function,
    aoi=[-5, 5],
    population_size=100,
    tournament_size=2,
    mutation_probability=0.05,
    crossover_probability=0.8,
    num_steps=50
)

def parameter_study():
    base_params = {
        "chromosome_length": 16,
        "obj_func_num_args": 2,
        "objective_function": objective_function,
        "aoi": [-5, 5],
        "population_size": 100,
        "tournament_size": 2,
        "mutation_probability": 0.05,
        "crossover_probability": 0.8,
        "num_steps": 50
    }
    num_runs = 7
    tests = {
        "mutation_probability": [0.01, 0.05, 0.1, 0.3],
        "tournament_size": [2, 5, 10, 20],
        "population_size": [20, 50, 100, 200],
        "crossover_probability": [0.4, 0.6, 0.8, 1.0]
    }
    for param_name, values in tests.items():
        print(f"\n=== {param_name} ===")
        for val in values:
            params = base_params.copy()
            params[param_name] = val
            results = []
            for _ in range(num_runs):
                ga = GeneticAlgorithm(**params)
                trace = ga.run()
                best_fitness = trace[-1][1]
                results.append(best_fitness)
            mean = np.mean(results)
            std = np.std(results)
            print(f"  {param_name}={val} -> mean={mean:.4f}, std={std:.4f}")
            ga = GeneticAlgorithm(**params)
            trace = ga.run()
            ga.plot_func(trace, title=f"Himmelblau - {param_name}={val}")
parameter_study()
