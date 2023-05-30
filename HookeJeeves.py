import numpy as np
from aa222_finalproject_regression import BuildModel, evaluateModelDesign, datasetLoad, evaluateModelProxy
from random import randint

#We already know the size of the first and last layer n
#We specify the MAX amount of layers we want but Hooke-Jeeves can vary that if it wants

def HookeJeeves(input_neurons, output_neurons, alpha, max_layers):
    '''Gives you optimal network structure from an initial guess'''
    #Hooking and Jeeving
    a = np.array([randint(0, 10) for p in range(0, max_layers + 2)])
    a[0] = input_neurons
    a[1] = randint(1, alpha)
    a[max_layers + 1] = output_neurons

    #After the first zero make every next value also 0 except for the last one
    idx = np.array(np.where(a == 0))

    if idx.size > 0:
        for i in range(idx[0, 0], len(a)-1):
            a[i] = 0

    a = np.array([28, 2, 0, 0, 0, 2])
    a_best = a
    x, y, xtest, ytest = datasetLoad()
    model = BuildModel(a, output_dim=2, activation_function='relu')
    # score2, KH = evaluateModelProxy(a, xtest, 100, output_dim=2)
    best_score = evaluateModelDesign(model, a, x, y, xtest, ytest, save=False, training_epochs=10)

    # #Then we start the loop
    while alpha > 1:
        improved = False

        #CHECK THE ENTIRE VECTOR
        for i in range(1, len(a)-1): #The range specifically does not change the first or the last value

            if a[i] < alpha:
                if a[i] == 0:
                    a[i] += alpha
                    model = BuildModel(a, output_dim=2, activation_function='relu')
                    score = evaluateModelDesign(model, a, x, y, xtest, ytest, save=False, training_epochs=10)
                    if score < best_score:
                        a_best = np.copy(a)
                        best_score = score
                        improved = True

                    a[i] -= alpha
                    break

                a[i] += alpha #Make sure we only have zeroes (maybe)
                model = BuildModel(a, output_dim=2, activation_function='relu')
                score = evaluateModelDesign(model, a, x, y, xtest, ytest, save=False, training_epochs=10)
                # score2, KH = evaluateModelProxy(a, xtest, 1000, output_dim=2)
                if score < best_score:
                    a_best = np.copy(a)
                    best_score = score
                    improved = True

                #Return to the a value
                a[i] -= alpha
            elif a[i] >= alpha:
                #Upper Branch
                a[i] += alpha
                model = BuildModel(a, output_dim=2, activation_function='relu')
                score = evaluateModelDesign(model, a, x, y, xtest, ytest, save=False, training_epochs=10)
                # score2, KH = evaluateModelProxy(a, xtest, 1000, output_dim=2)
                if score < best_score:
                    a_best = np.copy(a)
                    best_score = score
                    improved = True

                #Lower Branch
                a[i] -= 2*alpha
                model = BuildModel(a, output_dim=2, activation_function='relu')
                score = evaluateModelDesign(model, a, x, y, xtest, ytest, save=False, training_epochs=10)
                # score2, KH = evaluateModelProxy(a, xtest, 1000, output_dim=2)
                if score < best_score:
                    a_best = np.copy(a)
                    best_score = score
                    improved = True
                
                #Return to the a value
                a[i] += alpha

        a = a_best

        if improved == False:
            alpha = 0.5*alpha
            if alpha >= 1:
                alpha = round(alpha)
            else:
                break

    return a

    
        






