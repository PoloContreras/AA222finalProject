import numpy as np

import matplotlib.pyplot as plt
import h5py

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error

# import pdb

#LOADING A SPECIFIC DATASET INTO MATRIX FORM, USABLE BY NEURAL NETWORKS
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
def BuildModel(architecture,activation_function='relu'):

    input = architecture[0] #number of neurons matching input vector
    output_dim = architecture[-1] #number of neurons matching output vector

    #Quick verification that the input has SOMETHING
    if architecture[1] == 0:
        print("Error: the first layer is listed as having no neurons.")
        return None

    networkArch = [Dense(architecture[1],input_dim=input,activation=activation_function)] #initialization of the list describing network structure

    #Parsing the network architecture...
    maxLayers = len(architecture) #the length of the tuple is the maximum number of layers the generated network can have
    layers = 2 #index used to iterate through the tuple describing network structure (minimum of one layer plus input)
    while layers < maxLayers-1 and architecture[layers] != 0:
        networkArch += [Dense(architecture[layers],activation=activation_function)]
        layers += 1

    if layers < maxLayers-1 and any(architecture[layers+1:-1]):
        print("Error: a layer is listed as having nonzero neurons after a zero-neuron layer.")
        return None

    networkArch += [Dense(output_dim,activation='linear')] #output layer, with real-valued outputs

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

    for i in range(len(networkArch)): #give names to each of the layers in the model, for future reference
        model.layers[i]._name = 'layer_' + str(i)

    return model

#BUILD, TRAIN AND EVALUATE NEURAL NETWORK OF A SPECIFIC ARCHITECTURE
def evaluateModelDesign(model,architecture,x,y,xTest,yTest,save=True,training_epochs=10,verbose=True): #"model" is supposed to be the output of the BuildModel function, seen above 

    if training_epochs > 0: #training_epochs=0 will skip the training process and only evaluate the model performance on the test set, for a previously trained model
        model.fit(
            x, #training input data
            y, #training output labels for supervised learning
            epochs=training_epochs, #number of times to iterate over training data
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

    if verbose:
        print('Mean squared error of model '+str(architecture)+' on validation data: ',score) #the script will take your word for whatever the specified architecture was, so be careful

    if save:
        model.save('regression_finalproject_'+str(architecture)+'.h5') #save complete model as a .h5 file, for reference

    return score

def loadModel(architecture):
    model = keras.models.load_model('regression_finalproject_'+str(architecture)+'.h5') #this obviously won't work if there is no saved model in the folder with the specified architecture...
    return model

#EVALUATION FUNCTION FOR PREVIOUSLY SAVED MODEL, WITH NO OPTION TO TRAIN DNN FURTHER OR SAVE NETWORK AS A SEPARATE FILE
def evaluateModel(model,architecture,xTest,yTest,verbose=True):

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

    if verbose:
        print('Mean squared error of model with structure '+str(architecture)+' on validation data: ',score)

    return score

#SUBFUNCTION FOR PROXY SCORE CALCULATION: HAMMING DISTANCE
def distanceHamming(columnA,columnB): #function takes as input two binary column vectors
    return float(sum(columnA != columnB)) #Hamming distance is the total number of entries where the columns are different

#SUBFUNCTION FOR PROXY SCORE CALCULATION: BINARY VECTORS OF RELU ACTIVATIONS, USING MODEL, FOR EACH SAMPLE
def reLUvectors(architecture,model,xSamples):
    
    layers = len(architecture) - sum(tuple(x == y for x,y in zip(architecture,len(architecture)*(0,) ))) - 2 #number of layers in architecture that have more than zero neurons, excluding input

    C = np.zeros((np.shape(xSamples)[0],sum(architecture[1:-1]))) #binary row vectors: one column per neuron in model with ReLU, one row per sample input; ReLU will make entries either positive or turn negative elements to 0

    for i in range(layers):
        layer_output = model.get_layer('layer_'+str(i)).output #recall name of layer to review
        intermediate_model = keras.models.Model(inputs=model.input,outputs=layer_output) #intermediate model between model input and output we are reviewing
        intermediate_prediction = intermediate_model.predict(xSamples) 
        
        C[:,sum(architecture[1:i+1]):sum(architecture[1:i+2])] = intermediate_prediction

    C = 1.*(C > 0.) #conversion of activated outputs from all neurons into binary form

    return C.T

#PROXY SCORE FOR A NEURAL NETWORK OF A SPECIFIC ARCHITECTURE
def evaluateModelProxy(architecture,xTest,numSamples,output_dim=2):

    if numSamples > np.shape(xTest)[0]:
        print("Error: number of samples requested to be used for proxy exceeds number of samples in test dataset.")
        return None, None

    Na = sum(architecture[1:]) #total number of neurons in network, excluding input

    xSamples = np.random.default_rng().choice(xTest,numSamples,replace=False) #each row of this matrix is a sample from the test dataset to be used in proxy

    model = BuildModel(architecture) #the model is built and initialized, but not trained (also, this proxy function is designed for the ReLU activation function)

    binaryVectors = reLUvectors(architecture,model,xSamples) #binary vectors indicating where the ReLU was active, or not, for each neuron (one row per neuron) for each test example (one column per example)

    Kh = np.expand_dims(np.array([Na - distanceHamming(binaryVectors[:,0],binaryVectors[:,j]) for j in range(numSamples)]),0)
    for i in range(1,numSamples):
        Kh = np.vstack((Kh,np.expand_dims(np.array([Na - distanceHamming(binaryVectors[:,i],binaryVectors[:,j]) for j in range(numSamples)]),0)))

    (sign, logscore) = np.linalg.slogdet(Kh)
    proxyScore = logscore #for now, ignoring the sign of the determinant

    return proxyScore, Kh

#USING THE PROXY SCORE FUNCTION TO DEVELOP AN EXPLORATION DIRECTION IN HOOKE-JEEVES
def proxyStepDirection(Kh):

    step = -1*np.dot(np.linalg.pinv(Kh) , np.ones((np.shape(Kh)[1],1)))
    
    return step/np.linalg.norm(step)