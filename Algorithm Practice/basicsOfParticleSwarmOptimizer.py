#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 08:09:57 2025

@author: justice
"""

import numpy as np
import pandas as pd


def particle_swarm_optimization(cost_function, dimensions, bounds, num_particles, max_iter, w=0.5, phi_p=1.5, phi_g=1.5):
    """
    Perform Particle Swarm Optimization (PSO) to minimize the given cost function.

    Parameters:
    - cost_function: The objective function to minimize.
    - dimensions: Number of dimensions of the search space.
    - bounds: Tuple of lower and upper bounds for each dimension.
    - num_particles: Number of particles in the swarm.
    - max_iter: Maximum number of iterations.
    - w: Inertia weight.
    - phi_p: Cognitive coefficient.
    - phi_g: Social coefficient.

    Returns:
    - best_position: The best solution found.
    - best_value: The value of the cost function at the best solution.
    """

    # Initialize swarm
    lower_bound, upper_bound = bounds
    particles_position = np.random.uniform(lower_bound, upper_bound, (num_particles, dimensions))
    particles_velocity = np.random.uniform(-abs(upper_bound - lower_bound), abs(upper_bound - lower_bound), (num_particles, dimensions))


    # Initialize best known positions and swarm's global best
    #each particle has a personal best position
    personal_best_position = particles_position.copy()
    
    #each  position of a person  has  their best value
    personal_best_value = np.array([cost_function(p) for p in personal_best_position])
    
    #the global best position is the same as the best position of the amongs all the given positions
    global_best_position = personal_best_position[np.argmin(personal_best_value)]
    
    #out of this population bests we get the best  value
    global_best_value = np.min(personal_best_value)

    #we want to iterate a particular number of times 
    for iteration in range(max_iter): 
        #during this iteration or repeatetion we pick each and every  particle as an object 
        
        for i in range(num_particles):
            # we get some random influences that will affect the  our rate of movement which are social influences  and conitive influences Why: Adds stochastic behavior to avoid premature convergence
            # Update velocity
            r_p = np.random.uniform(0, 1, dimensions)
            r_g = np.random.uniform(0, 1, dimensions)
            particles_velocity[i] = (
                w * particles_velocity[i]
                + phi_p * r_p * (personal_best_position[i] - particles_position[i])
                + phi_g * r_g * (global_best_position - particles_position[i])
            )

            # Update position of this particular particle
            particles_position[i] += particles_velocity[i]

            # Apply bounds to position let the particles new postion be between the search space
            particles_position[i] = np.clip(particles_position[i], lower_bound, upper_bound)

            # Evaluate the new position get the current cost at that position supposed you want to get to the optimal position
            current_value = cost_function(particles_position[i])

            # Update personal best now we  update the table that contains this values and then personal best, personal position  and we set the global best position to this personal best if its less than it 
            if current_value < personal_best_value[i]:
                personal_best_position[i] = particles_position[i]
                personal_best_value[i] = current_value

                # Update global best
                if current_value < global_best_value:
                    global_best_position = particles_position[i]
                    global_best_value = current_value

        print(f"Iteration {iteration+1}/{max_iter}: Best Value = {global_best_value}")
        #keep doing this for a while to see what the optimal positions are 
    return global_best_position, global_best_value



def sphere_function(x):
    return sum(x**2)

dimensions = 2
bounds = (-10, 10)
num_particles = 30
max_iter = 100

best_position, best_value = particle_swarm_optimization(
    cost_function=sphere_function,
    dimensions=dimensions,
    bounds=bounds,
    num_particles=num_particles,
    max_iter=max_iter
)
print("Best Position:", best_position)
print("Best Value:", best_value)