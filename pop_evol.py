import numpy as np

import random

from aa222_finalproject_regression import BuildModel, evaluateModel, datasetLoad
from HookeJeeves import HookeJeevesPop
from random import randint


# max neurons is kinda arbitrary, so look into manipulating that maybe
def generate_random_architecture(max_layers=5, max_neurons=64):
   num_layers = random.randint(2, max_layers)
   architecture = [28]  # Starting with input layer neurons

   for i in range(num_layers - 1):
       if i >=2 and architecture[i-1] == 0:
        architecture.extend([0]*(num_layers -1-i))
        break
       architecture.append(random.randint(0, max_neurons))
  
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
    scores = []
    for architecture in population:
        if has_zero_neurons_after_layer(architecture):
            scores.append(0)  # Assign a score of 0 to architectures with invalid configurations
        else:
            model = BuildModel(architecture)
            score = evaluateModel(model, architecture, xTest, yTest, verbose=False)
            scores.append(score)
    return scores

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
  
   for _ in range(num_mutations):
       # Generate a mutated architecture using Hooke-Jeeves
       mutated_architecture_np = HookeJeevesPop(architecture)
       mutated_architecture_tuple= tuple(mutated_architecture_np)

       if mutated_architecture_tuple[0] == 0:
            continue
    

       # Add the mutated architecture to the list
       mutated_architecture.extend(mutated_architecture_tuple)


   return tuple(mutated_architecture)


# Main (Regression Dataset)
x, y, xTest, yTest = datasetLoad()
population_size = 50
parents_max = 10
generations_max = 10


# generations_max can definitely be increased fo’ sho’


population = generate_population(population_size)


for generation in range(generations_max):
   scores = evaluate_population(population, xTest, yTest)
   best_arch = [population[i] for i in np.argsort(scores)[:parents_max]]
   best_networks = []


   for architecture in best_arch:
       best_networks.append(HookeJeevesPop(architecture))


   population = evolve_population(population, scores, parents_max)


# Get best architecture and best network
best_architecture = population[np.argmin(scores)]
best_network = BuildModel(best_architecture)
best_network_output = evaluateModel(best_network, best_architecture, xTest, yTest, verbose = True)

# Get the Ensemble Working
ensemble_outputs = []
for network, architecture in zip(best_networks, best_arch):
    ensemble_outputs.append(evaluateModel(network, architecture, xTest, yTest))

ensemble_output = np.mean(ensemble_outputs)


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







