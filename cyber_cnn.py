from tkinter import Y
import numpy as np

import mnist
import cyberData
from src.activation import relu, softmax, softmaxConv
from src.cost import softmax_cross_entropy, softmax_conv_cross_entropy
from src.layers.conv import Conv
from src.layers.dropout import Dropout
from src.layers.fc import FullyConnected
from src.layers.flatten import Flatten
from src.layers.pool import Pool
from src.nn import NeuralNetwork
from src.optimizer import adam

isLastLayerConv = False

def one_hot(x, num_classes=10):
    print(x.shape)
    out = np.zeros((x.shape[0], num_classes))
    out[np.arange(x.shape[0]), x[:, 0]] = 1
    return out


y_train = np.array(([0,1,2,3,4]))
def one_hot_Conv(y_train, output_kernel_size, num_classes=10):
    out = np.zeros((y_train.shape[0],output_kernel_size,output_kernel_size,num_classes))
    for i in range(y_train.shape[0]):
        counting = 0
        for k1 in range(output_kernel_size):
            for k2 in range(output_kernel_size):
                out[i,k1,k2,y_train[i]] = 1
                counting += 1
    return out
one_hot_Conv(y_train,1,5).shape

def preprocess(x_train, y_train, x_test, y_test, x_predict):
    global isLastLayerConv
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype(np.float32)
    if(isLastLayerConv == False):
        y_train = one_hot(y_train.reshape(y_train.shape[0], 1),5)
    if(isLastLayerConv == True):
        y_train = one_hot_Conv(y_train,1,5)
    x_train /= 255
    x_test /= 255
    return x_train, y_train, x_test, y_test, x_predict
   
def ancientWays():
    global isLastLayerConv
    isLastLayerConv = False
    x_train, y_train, x_test, y_test, x_predict = preprocess(*cyberData.load())
    y_train
    cnn = NeuralNetwork(
        input_dim=(28, 28, 1),
        layers=[
            Conv(5, 1, 32, activation=relu),
            Pool(2, 2, 'max'),
            Dropout(0.75),
            Flatten(),
            FullyConnected(128, relu),
            Dropout(0.9),
            FullyConnected(5, softmax),
        ],
        cost_function=softmax_cross_entropy,
        optimizer=adam
    )
    
    cnn.train(x_train, y_train,
              mini_batch_size=x_train.shape[0],
              learning_rate=0.001,
              num_epochs=1,
              validation_data=(x_test, y_test))


def newWays():
    global isLastLayerConv
    isLastLayerConv = True
    x_train, y_train, x_test, y_test, x_predict = preprocess(*cyberData.load())
    input_dim=(28, 28, 1)
    layers =[
            Conv(5, 1, 32, activation=relu),
            Pool(2, 2, 'max'),
            Dropout(0.75),
            Conv(12, 1, 128, activation=relu),
            Dropout(0.7),
            Conv(1, 1, y_train.shape[0], activation=softmaxConv),
        ]
    cnn = NeuralNetwork(
        input_dim=input_dim,
        layers=[
            Conv(14, 14, 32, activation=relu),
            Pool(2, 2, 'max'),
            Dropout(0.75),
            Conv(1, 1, 128, activation=relu),
            Dropout(0.7),
            Conv(1, 1, y_train.shape[0], activation=softmaxConv),
        ],
        cost_function=softmax_conv_cross_entropy,
        optimizer=adam,
        saveWeights=False,
        loadWeights=True
    )

    #cnn.train(x_train, y_train,
    #          mini_batch_size=x_train.shape[0],
    #          learning_rate=0.001,
    #          num_epochs=20,
    #          validation_data=(x_test, y_test))
    
    input_dim=(28,x_train.shape[0] * 28,1)
    cnn.slidingWindowsDimensionsUpdate(input_dim)
    predictions = cnn.predict(x_predict.astype(np.float32))
    print(np.argmax(predictions, axis=3))
if __name__ == "__main__":
    #ancientWays()
    newWays()


##Todo: reshape x_train properly