import numpy as np

import matplotlib.pyplot as plt
import h5py

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error

# import pdb

def datasetLoad(inputSteps=15,traindata='pedestrian_traindata_ohio.hdf5',testdata='pedestrian_testdata_ohio.hdf5'):

    #inputSteps is the number of time steps into which the trajectory data is divided: the first inputSteps-1 time steps are used as an input and the next time step is the prediction

    f = h5py.File(traindata,'r')
    g = h5py.File(testdata,'r')

    k = f['trajectories'][()] - f['failures'][()] #number of successful trajectories, to be turned into training data pairs
    print('Training model based on '+str(k)+' full trajectories.')

    trajectorySteps = int(f['time'][()]*f['frequency'][()] + 1) #number of time steps in a complete trajectory (default: 151)

    generalFirst = True #index to determine if current data point is the first being used
    for i in range(k): #add data to input/output pairs for each trajectory in the dataset
        # pdb.set_trace()
        for j in range(trajectorySteps-inputSteps): #add input/output pairs
            if generalFirst:
                x = f['pedestrian_states'][j:j+inputSteps-1,:,i].flatten()
                y = f['pedestrian_states'][j+inputSteps,:,i]
                generalFirst = False
            else:
                x = np.vstack((x,f['pedestrian_states'][j:j+inputSteps-1,:,i].flatten()))
                y = np.vstack((y,f['pedestrian_states'][j+inputSteps,:,i]))

    print("input tensor shape (training): ",x.shape)
    print("output label matrix shape (training): ",y.shape)

    kTest = g['trajectories'][()] - g['failures'][()] #number of successful trajectories, to be turned into training data pairs
    print('Validating models with '+str(kTest)+' full trajectories.')

    for i in range(kTest): #add data to input/output pairs for each trajectory in the dataset
        for j in range(trajectorySteps-inputSteps): #add input/output pairs
            if i==0 and j ==0:
                xTest = g['pedestrian_states'][j:j+inputSteps-1,:,i].flatten()
                yTest = g['pedestrian_states'][j+inputSteps,:,i]
            else:
                xTest = np.vstack((xTest,g['pedestrian_states'][j:j+inputSteps-1,:,i].flatten()))
                yTest = np.vstack((yTest,g['pedestrian_states'][j+inputSteps,:,i]))

    print("input tensor shape (validation): ",xTest.shape)
    print("output label matrix shape (validation): ",yTest.shape)

    return x,y,xTest,yTest


#DEFINIITION OF NEURAL NETWORK STRUCTURE
def BuildModel(architecture,input=28,activation_function='relu'):

    #Quick verification that the input has SOMETHING
    if architecture[0] == 0:
        print("Error: the first layer is listed as having no neurons.")
        return None

    networkArch = [Dense(architecture[0],input_dim=input,activation=activation_function)] #initialization of the list describing network structure

    #Parsing the network architecture...
    maxLayers = len(architecture) #the length of the tuple is the maximum number of layers the generated network can have
    layers = 1 #index used to iterate through the tuple describing network structure
    while architecture[layers] != 0:
        networkArch += [Dense(architecture[layers],input_dim=input,activation=activation_function)]
        layers += 1

    if any(networkArch[layers:]):
        print("Error: a layer is listed as having nonzero neurons after a zero-neuron layer.")
        return None

    # Example
    # model = Sequential([
    #     Dense(32,input_dim =28, activation=activation_function),
    #     Dense(32,activation=activation_function),
    #     Dense(2,activation='linear'), #two real system outputs 
    # ])
    model = Sequential(networkArch) #definition of a sequential model with the layers described by the input tuple

    model.compile(
        'adam', #Adam optimizer, for simplicity
        loss ='mean_squared_error',
        #metrics=['mean_squared_error'], #include this metric?
    )

    return model

def evaluateModelDesign(architecture,x,y,xTest,yTest,save=True,input=28,activation_function='relu'):

    model = BuildModel(architecture,input,activation_function)

    model.fit(
        x, #training input data
        y, #training output labels for supervised learning
        epochs=10,
        validation_data =(xTest,yTest), #test set for model verification
    )

    # y_krm = model.predict(x)

    # mse_krm = mean_squared_error(y,y_krm)
    # print(mse_krm)

    # plt.plot(y,label='original y')
    # plt.plot(y_krm, label='predicted y')
    # plt.legend()
    # plt.show()

    # plt.plot(yTest,label='original y (validation)')
    # plt.plot(model.predict(xTest),label='predicted y (validation)')
    # plt.legend()
    # plt.show()

    score = mean_squared_error(yTest,model.predict(xTest))

    print('Mean squared error of model '+str(architecture)+' on validation data: ',score)

    if save:
        model.save('regression_ensemble_'+str(architecture)+'.h5') #save complete model as a .h5 file, for reference

    return score