import numpy as np
from aa222_finalproject_regression import BuildModel, evaluateModelDesign, datasetLoad, evaluateModelProxy, evaluateModelProxyAlt, mnistLoad
from random import randint
from scipy import linalg
import math as math

#We already know the size of the first and last layer n
#We specify the MAX amount of layers we want but Hooke-Jeeves can vary that if it wants

def HookeJeeves(input_neurons, output_neurons, alpha, max_layers):
    '''Gives you optimal network structure from an initial guess'''
    #Hooking and Jeeving
    score_hist = []
    proxy_score_hist = []
    ahist = []

    a = np.array([randint(0, 10) for p in range(0, max_layers + 2)])
    a[0] = input_neurons
    a[1] = randint(1, alpha)
    a[max_layers + 1] = output_neurons
    a = np.array([28, 10, 10, 10, 10, 10, 2])

    #After the first zero make every next value also 0 except for the last one
    idx = np.array(np.where(a == 0))

    if idx.size > 0:
        for i in range(idx[0, 0], len(a)-1):
            a[i] = 0

    a_best = a
    x, y, xtest, ytest = datasetLoad()
    # x, y, xtest, ytest = mnistLoad()
    model = BuildModel(a, activation_function='relu', regression=True)
    best_score = evaluateModelDesign(model, a, x, y, xtest, ytest, save=False, training_epochs=10, regression=True)
    # best_score, KH = evaluateModelProxyAlt(a, xtest, 80)


    # #Then we start the loop
    while alpha >= 1:
        improved = False
        score_hist.append(best_score)
        ahist.append(a_best.tolist())

        #CHECK THE ENTIRE VECTOR
        for i in range(1, len(a)-1): #The range specifically does not change the first or the last value
 
            if a[i] <= 2*alpha:
                if a[i] == 0:
                    a[i] += alpha
                    model = BuildModel(a, activation_function='relu', regression=True)
                    score = evaluateModelDesign(model, a, x, y, xtest, ytest, save=False, training_epochs=10, regression=True)
                    # score, KH = evaluateModelProxyAlt(a, xtest, 80)
                    if score < best_score:
                        a_best = np.copy(a)
                        best_score = score
                        improved = True
                        a[i] -= alpha

                    break

                a[i] += alpha
                model = BuildModel(a, activation_function='relu', regression=True)
                score = evaluateModelDesign(model, a, x, y, xtest, ytest, save=False, training_epochs=10, regression=True)
                # score, KH = evaluateModelProxyAlt(a, xtest, 80)
                if score < best_score:
                    a_best = np.copy(a)
                    best_score = score
                    improved = True

                a[i] -= alpha

            elif a[i] > 2*alpha:
                #Upper Branch
                a[i] += alpha
                model = BuildModel(a, activation_function='relu', regression=True)
                score = evaluateModelDesign(model, a, x, y, xtest, ytest, save=False, training_epochs=10, regression=True)
                # score, KH = evaluateModelProxyAlt(a, xtest, 80)
                if score < best_score:
                    a_best = np.copy(a)
                    best_score = score
                    improved = True

                #Lower Branch
                a[i] -= 2*alpha
                model = BuildModel(a, activation_function='relu', regression=True)
                score = evaluateModelDesign(model, a, x, y, xtest, ytest, save=False, training_epochs=10, regression=True)
                # score, KH = evaluateModelProxyAlt(a, xtest, 80)
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

    return a, score_hist, ahist, proxy_score_hist

def PosDefMatrix(max_layers):
    A = np.array(np.random.randint(10, size = (max_layers,max_layers)))
    B = np.multiply(A.transpose(), A)
    U, sigma, V = linalg.svd(B)
    U = np.around(U)

    #Adding the additional search direction
    s2 = np.sum(U,axis=1)/math.sqrt(max_layers)
    s2 = np.around(s2)
    s2.shape = (max_layers, 1)
    U = np.append(U, s2, axis=1)

    #Add zeroes to index 0 and end of each vector to make sure we do not change the input or output dimensions
    U = np.pad(U, ([1, 1], [0, 0]), mode='constant')
    U = U.astype(int)

    return U


def HookeJeevesSVD(input_neurons, output_neurons, alpha, max_layers):
    '''Gives you optimal network structure from an initial guess'''
    score_hist = []
    proxy_score_hist = []
    ahist = []

    #Hooking and Jeeving
    a = np.array([randint(0, 10) for p in range(0, max_layers + 2)])
    a[0] = input_neurons
    a[1] = randint(1, alpha)
    a[max_layers + 1] = output_neurons
    a = np.array([28, 10, 10, 10, 10, 10, 2])

    #After the first zero make every next value also 0 except for the last one
    idx = np.array(np.where(a == 0))

    if idx.size > 0:
        for i in range(idx[0, 0], len(a)-1):
            a[i] = 0

    a_best = a
    x, y, xtest, ytest = datasetLoad()
    # x, y, xtest, ytest = mnistLoad()
    model = BuildModel(a, activation_function='relu', regression=True)
    # best_score, KH = evaluateModelProxy(a, xtest, 80)
    best_score = evaluateModelDesign(model, a, x, y, xtest, ytest, save=False, training_epochs=10)


    # Then we start the loop
    while alpha >= 1:
        score_hist.append(best_score)
        ahist.append(a_best.tolist())
        U = PosDefMatrix(max_layers)
        search = alpha*U
        improved = False

        #CHECK THE ENTIRE VECTOR
        for i in range(max_layers): #The range specifically does not change the first or the last value

            step = search[:, i]
            a += step
            #Replace all negatives with 0
            a[a<0] = 0

            idx = np.array(np.where(a == 0))
            if idx.size > 0:
                for i in range(idx[0, 0], len(a)-1):
                    a[i] = 0
            
            if a[1] == 0:
                a[1] = alpha

            model = BuildModel(a, activation_function='relu', regression=True)
            score = evaluateModelDesign(model, a, x, y, xtest, ytest, save=False, training_epochs=10)
            # score, KH = evaluateModelProxyAlt(a, xtest, 80)
            if score < best_score:
                best_score = score
                a_best = a
                improved = True
            
            a -= step

            
        a = a_best
        search = U

        if improved == False:
            alpha = 0.5*alpha
            if alpha >= 1:
                alpha = round(alpha)
            else:
                break

    return a, score_hist

def HookeJeevesPop(a, alpha=10):
    '''Gives you optimal network structure from an initial guess'''
    #Hooking and Jeeving
    max_layers = len(a) - 2

    #After the first zero make every next value also 0 except for the last one
    idx = np.array(np.where(a == 0))

    if idx.size > 0:
        for i in range(idx[0, 0], len(a)-1):
            a[i] = 0

    a_best = a
    x, y, xtest, ytest = datasetLoad()
    model = BuildModel(a, activation_function='relu')
    best_score, KH = evaluateModelProxy(a, xtest, 80)
    # best_score = evaluateModelDesign(model, a, x, y, xtest, ytest, save=False, training_epochs=10)


    # Then we start the loop
    while alpha >= 1:
        U = PosDefMatrix(max_layers)
        search = alpha*U
        improved = False

        #CHECK THE ENTIRE VECTOR
        for i in range(max_layers): #The range specifically does not change the first or the last value

            step = search[:, i]
            a += step
            #Replace all negatives with 0
            a[a<0] = 0

            idx = np.array(np.where(a == 0))
            if idx.size > 0:
                for i in range(idx[0, 0], len(a)-1):
                    a[i] = 0


            if a[1] == 0:
                a[1] = alpha

            # score = evaluateModelDesign(model, a, x, y, xtest, ytest, save=False, training_epochs=10)
            score, KH = evaluateModelProxy(a, xtest, 80)
            if score < best_score:
                best_score = score
                a_best = a
                improved = True
            
            a -= step

            
        a = a_best
        search = U

        if improved == False:
            alpha = 0.5*alpha
            if alpha >= 1:
                alpha = round(alpha)
            else:
                break

    return a
    
        






