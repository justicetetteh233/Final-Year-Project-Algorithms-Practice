import numpy as np
import pandas as pd


def eosa(obj_func, num_variables, num_agents, max_iter, lb, ub):
    # Initialize population
    population = np.random.rand(num_agents, num_variables) * (ub - lb) + lb
    
    # this is what i added to see the population  @justice
    population_df = pd.DataFrame(population, columns=[f"Var_{i+1}" for i in range(num_variables)])
    print(population_df)
    print(population[0])
    #@endJustice

    
    fitness = np.zeros(num_agents)
    
    #@justice
    print(fitness)
    #@endJustice
    
    # Evaluate fitness of each agent
    for i in range(num_agents):
        fitness[i] = obj_func(population[i, :])
    
    #@justice what happens after this loop
    print(fitness)
    #@endJustice
    
    # Main loop
    for iter in range(max_iter):
        # Update position and fitness of each agent
        for i in range(num_agents):
            # Select a random agent as the transmitter
            transmitter_index = np.random.randint(num_agents)
            #@justice
            print(f"current transmitter {transmitter_index}")
            #@justice
            while transmitter_index == i:
                #@justice
                print('the current transmitter is this agent so look for another agent')
                #@endJustice
                
                transmitter_index = np.random.randint(num_agents)
            
            # Update the position of the agent based on the transmission dynamics
            #@justice
            print(f"this is the current structure of the agent   {population[i, :]}")
            #@endJustice
            
            new_solution = population[i, :] + np.random.randn(num_variables) * (population[transmitter_index, :] - population[i, :])
            #@justice
            #print(f"this is the new structure of the agent   {new_solution[i, :]}")
            #@endJustice
            
            # Clip new solution to ensure it stays within bounds
            new_solution = np.clip(new_solution, lb, ub)
            #@justice
            print(f"let this new solution match the boundaries {new_solution}")
            #@endJustice
            
            # Evaluate fitness of the new solution
            new_fitness = obj_func(new_solution)
            #@justice
            print(f"what is the current fitness of this agent based on its varables {new_fitness}")
            #@endJustice
            
            # Update if the new solution is better
            if new_fitness < fitness[i]:
                population[i, :] = new_solution
                print(f"after this attack this agent at {i} is strong so we will update the  population table and update his fitness {new_fitness}")
                fitness[i] = new_fitness
    
    # Find the best solution in the final population
    best_fitness = np.min(fitness)
    best_index = np.argmin(fitness)
    best_solution = population[best_index, :]
    
    return best_solution, best_fitness


# Example of an objective function (e.g., Sphere function)
def obj_func(x):
    return np.sum(x**2)

# Set parameters
num_variables = 10
num_agents = 50
max_iter = 100
lb = -5 * np.ones(num_variables)
ub = 5 * np.ones(num_variables)

# Call the eosa function
best_solution, best_fitness = eosa(obj_func, num_variables, num_agents, max_iter, lb, ub)

print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)

