import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Generate a simple dataset for feature selection
np.random.seed(42)
X, y = make_classification(n_samples=200, n_features=20, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Fitness function (evaluate subsets of features)
def fitness_function(selected_features):
    if not any(selected_features):
        return 0  # Avoid selecting no features
    
    # Filter features based on selection
    X_train_subset = X_train[:, selected_features]
    X_test_subset = X_test[:, selected_features]

    # Train a classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_subset, y_train)
    
    # Return the accuracy as fitness
    y_pred = clf.predict(X_test_subset)
    return accuracy_score(y_test, y_pred)

# Step 3: Initialize EOSA parameters
population_size = 20
num_features = X.shape[1]
num_generations = 50
infection_rate = 0.3
mutation_rate = 0.1

# Initialize population (random feature subsets)
population = np.random.randint(0, 2, (population_size, num_features))
fitness_scores = np.zeros(population_size)

# Step 4: EOSA main loop
best_solution = None
best_fitness = 0
history = []

for generation in range(num_generations):
    # Evaluate fitness for the population
    for i in range(population_size):
        fitness_scores[i] = fitness_function(population[i])

    # Track the best solution
    gen_best_idx = np.argmax(fitness_scores)
    if fitness_scores[gen_best_idx] > best_fitness:
        best_fitness = fitness_scores[gen_best_idx]
        best_solution = population[gen_best_idx].copy()

    history.append(best_fitness)

    # Infection dynamics: Spread best solutions to weaker ones
    for i in range(population_size):
        if np.random.rand() < infection_rate:
            # Replace weaker solutions with parts of the best
            population[i] = np.where(np.random.rand(num_features) < 0.5, best_solution, population[i])

    # Mutation: Randomly toggle some bits
    for i in range(population_size):
        if np.random.rand() < mutation_rate:
            mutation_points = np.random.randint(0, num_features, int(num_features * 0.1))
            population[i, mutation_points] = 1 - population[i, mutation_points]

# Step 5: Visualization
plt.plot(history)
plt.title('Convergence of EOSA')
plt.xlabel('Generation')
plt.ylabel('Best Fitness (Accuracy)')
plt.show()

# Step 6: Display results
print("Best Fitness:", best_fitness)
print("Selected Features:", np.where(best_solution == 1)[0])
