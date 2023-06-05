import numpy as np

import random

import pandas as pd

import matplotlib.pyplot as plt

import multiprocessing

from aa222_finalproject_regression import BuildModel, evaluateModel, datasetLoad
from HookeJeeves import HookeJeevesPop
from random import randint


# max neurons is kinda arbitrary, so look into manipulating that maybe
def generate_random_architecture(max_layers=5, max_neurons=64):
    num_layers = random.randint(2, max_layers)
    architecture = [28]  # Starting with input layer neurons

    for i in range(num_layers - 1):
        if i >= 1 and architecture[i-1] == 0:
            architecture.extend([0] * (num_layers - i))
            break
        architecture.append(random.randint(1, max_neurons))

    if architecture[1] == 0 or (num_layers >= 3 and architecture[2] == 0):
        architecture[3:-1] = [0] * (len(architecture) - 4)

    architecture[-1] = 2

    return tuple(architecture)

def generate_population(size):
    population = []
    unique_architectures = set()
    while len(population) < size:
        architecture = generate_random_architecture()
        architecture_tuple = tuple(architecture)
        if architecture_tuple not in unique_architectures:
            population.append(architecture_tuple)
            unique_architectures.add(architecture_tuple)
    return population


def evaluate_population(population, xTest, yTest):
    pool = multiprocessing.Pool()
    scores = pool.map(evaluate_architecture, [(arch, xTest, yTest) for arch in population])
    pool.close()
    pool.join()
    return scores

def evaluate_architecture(args):
    architecture, xTest, yTest = args
    if has_zero_neurons_after_layer(architecture):
        return 0
    else:
        model = BuildModel(architecture)
        score = evaluateModel(model, architecture, xTest, yTest, verbose=False)
        return score


def has_zero_neurons_after_layer(architecture):
    for i in range(1, len(architecture)):
        if architecture[i-1] == 0 and architecture[i] != 0:
            return True
    return False

def evolve_population(population, scores, parents_max):
   sorted_indices = np.argsort(scores)
   parents = [population[i] for i in sorted_indices[:parents_max]]
   new_pop = parents.copy()


   while len(new_pop) < len(population):
       parent_random = random.choice(parents)
       child_random = mutate_architecture(parent_random)
       new_pop.append(child_random)


   return new_pop

def mutate_architecture(architecture, num_mutations=1):
    mutated_architecture = list(architecture)
    
    # Generate a mutated architecture using Hooke-Jeeves
    mutated_architecture_np = HookeJeevesPop(architecture)
    mutated_architecture_tuple = tuple(mutated_architecture_np)
    
    if mutated_architecture_tuple[0] == 0:
        return architecture
    
    if mutated_architecture_tuple[1] == 0 or (len(mutated_architecture_tuple) >= 3 and mutated_architecture_tuple[2] == 0):
        mutated_architecture_tuple = list(mutated_architecture_tuple)
        mutated_architecture_tuple[3:-1] = [0] * (len(mutated_architecture_tuple) - 4)
        mutated_architecture_tuple = tuple(mutated_architecture_tuple)
    
    # Apply additional mutations
    for _ in range(num_mutations - 1):
        # Randomly select a position in the architecture to mutate
        mutation_index = random.randint(0, len(mutated_architecture) - 1)
        
        # Ensure a maximum of five layers
        if mutation_index == 1:
            mutated_architecture[mutation_index] = random.randint(1, 64)
        elif mutation_index >= 2 and mutation_index < 6:
            mutated_architecture[mutation_index] = random.randint(0, 64)
        
        # Ensure the last layer is equal to 2
        if mutation_index == len(mutated_architecture) - 1:
            mutated_architecture[mutation_index] = 2
    
    return tuple(mutated_architecture)




# Main (Regression Dataset)
if __name__ == '__main__':
    x, y, xTest, yTest = datasetLoad()
    population_size = 5
    parents_max = 3
    generations_max = 3
    # generations_max can definitely be increased fo’ sho’

    population = generate_population(population_size)

    scores_history = []
    generations = []

    for generation in range(generations_max):
        scores = evaluate_population(population, xTest, yTest)
        best_arch = [population[i] for i in np.argsort(scores)[:parents_max]]
        best_networks = []

        population = evolve_population(population, scores, parents_max)

        scores_history.append(scores)
        generations.append(generation)

    # Get best architecture and best network
    best_architecture = population[np.argmin(scores)]
    best_network = BuildModel(best_architecture)
    best_network_output = evaluateModel(best_network, best_architecture, xTest, yTest, verbose = True)

    # Get the Ensemble Working
    ensemble_outputs = []
    for architecture in best_arch:
        network = BuildModel(architecture)
        ensemble_outputs.append(evaluateModel(network, architecture, xTest, yTest))

    ensemble_output = np.mean(ensemble_outputs)
    # Create a DataFrame with architectures and scores
    data = {
        'Architecture': population,
        'Score': scores
    }
    df = pd.DataFrame(data)

    # Display the table
    print(df)

  
   

    for i in range(len(scores_history)):
        plt.plot(generations, scores_history[i], marker='o', label='Generation {}'.format(i + 1))
      
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.title('Scores over Generations')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    data = {
        'Architecture': best_arch,
        'Output': ensemble_outputs
    }
    df = pd.DataFrame(data)

    # Display the table
    print(df)

    # Create a box plot
    plt.boxplot(ensemble_outputs)
    plt.xlabel('Network')
    plt.ylabel('Output')
    plt.title('Ensemble Outputs')
    plt.xticks(range(1, len(best_arch) + 1), best_arch, rotation=90)
    plt.tight_layout()
    plt.show()

# —--------------------------------------------------
# **Potential code for best initial guess before hooke-Jeeves implementation**

# def random_search(iterations):
#    best_architecture = None
#    best_score = float('inf')


#    for _ in range(iterations):
#        architecture = generate_random_architecture()
#        model = BuildModel(architecture)
#        score = evaluateModel(model, architecture, xTest, yTest, verbose=False)


#        if score < best_score:
#            best_score = score
#            best_architecture = architecture


#    return best_architecture


# iterations = 100 
# best_guess = random_search(iterations)







