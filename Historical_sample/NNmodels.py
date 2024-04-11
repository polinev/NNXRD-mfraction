# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:07:23 2022

@author: PURUSHOT

NN models for Victor dataset

"""
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

metricsNN = [
            keras.metrics.FalseNegatives(name="fn"),
            keras.metrics.FalsePositives(name="fp"),
            keras.metrics.TrueNegatives(name="tn"),
            keras.metrics.TruePositives(name="tp"),
            ]

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

# Only DNN model

def DNN_model(  shape, 
                layer_activation="relu", 
                output_activation="relu",
                dropout=0.3,
                DNN_layers = 3,
                DNN_filters = [5000,2000,500],
                output_neurons = 11,
                learning_rate=0.001,
                optimizer="adam",
                loss="mae"):
    
    model = Sequential()
    model.add(keras.Input(shape=(int(shape),)))
    ## Hidden layers
    # assert len(DNN_filters) == DNN_layers, "Number of DNN layers and filter length doesnt match"
    for lay in range(DNN_layers):
        model.add(Dense(int(DNN_filters[lay])))
        model.add(Activation(layer_activation))
        if dropout > 0:
            model.add(Dropout(dropout))
    ## Output layer 
    if output_activation != None:
        model.add(Dense(int(output_neurons), activation=output_activation))
    else:
        model.add(Dense(int(output_neurons)))
    ## Compile model
    if optimizer == "adam":
        otp = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        otp = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        
    model.compile(loss=loss, optimizer=otp, metrics=['mse', 'mae'])
    #model.compile(loss=loss, optimizer=otp)
    return model

# CNN + DNN Model to predict phase fraction and position
def CNN_DNN_model(shape, 
                 layer_activation="relu", 
                 output_activation="relu",
                 dropout=0.3,
                 stride = [2,2,2],
                 kernel_size = [5,5,5],
                 pool_size=[2,2,2],
                 CNN_layers = 3,
                 CNN_filters = [64,128,256],
                 DNN_layers = 3,
                 DNN_filters = [5000,2000,500],
                 output_neurons = 11,
                 learning_rate = 0.001,
                 optimizer="adam",
                 loss = 'mae',
                 predict='pp_h'):
    #predict can be # pp, h, tc or pp_h, pp_tc, h_tc, pp_h_tc
    
    inputs = keras.layers.Input(shape, name="InputLayer")
    
    # assert len(CNN_filters) == CNN_layers, "Number of CNN layers and filter length doesnt match"
    for lay in range(CNN_layers):
        if lay == 0:
            conv1 = keras.layers.Conv1D(filters=CNN_filters[lay], kernel_size=kernel_size[lay], 
                                        strides=stride[lay], 
                                        activation=layer_activation, name="Conv_"+str(lay+1))(inputs)
            pool1 = keras.layers.MaxPooling1D(pool_size=pool_size[lay], \
                                              name="Pool_"+str(lay+1))(conv1)
        else:
            conv1 = keras.layers.Conv1D(filters=CNN_filters[lay], kernel_size=kernel_size[lay], 
                                        strides=stride[lay], 
                                        activation=layer_activation, name="Conv_"+str(lay+1))(pool1)
            pool1 = keras.layers.MaxPooling1D(pool_size=pool_size[lay], \
                                              name="Pool_"+str(lay+1))(conv1)
    flatten = keras.layers.Flatten(name="Flatten")(pool1)

    # DNN for fraction prediction
    # assert len(DNN_filters) == DNN_layers, "Number of DNN layers and filter length doesnt match"
    if 'pp' in predict:
        for lay in range(DNN_layers):
            if lay == 0:
                ppKL = keras.layers.Dense(DNN_filters[lay], activation=layer_activation, \
                                          name="pp_Dense_"+str(lay+1))(flatten)
                ppKL = keras.layers.Dropout(dropout, name="pp_Dropout"+str(lay+1))(ppKL)   
            else:
                
                ppKL = keras.layers.Dense(DNN_filters[lay], activation=layer_activation, \
                                          name="pp_Dense_"+str(lay+1))(ppKL)
                ppKL = keras.layers.Dropout(dropout, name="pp_Dropout"+str(lay+1))(ppKL) 
        ## Output layer 
        if output_activation != None:
            ppKL = keras.layers.Dense(output_neurons, activation=output_activation, name="pp_Dense_out")(ppKL)
        else:
            ppKL = keras.layers.Dense(output_neurons, name="pp_Dense_out")(ppKL)
    
    if 'h' in predict:
        # DNN for phase position prediction
        for lay in range(DNN_layers):
            if lay == 0:
                hKL = keras.layers.Dense(DNN_filters[lay], activation=layer_activation, \
                                          name="h_Dense_"+str(lay+1))(flatten)
                hKL = keras.layers.Dropout(dropout, name="h_Dropout"+str(lay+1))(hKL)   
            else:
                hKL = keras.layers.Dense(DNN_filters[lay], activation=layer_activation, \
                                          name="h_Dense_"+str(lay+1))(hKL)
                hKL = keras.layers.Dropout(dropout, name="h_Dropout"+str(lay+1))(hKL) 
        ## Output layer 
        if output_activation != None:
            hKL = keras.layers.Dense(output_neurons, activation=output_activation, name="h_Dense_out")(hKL)
        else:
            hKL = keras.layers.Dense(output_neurons, name="h_Dense_out")(hKL)
    
    # Common output
    if 'pp' in predict and 'h' in predict:
        outputs = keras.layers.concatenate([ppKL, hKL], axis=1, name='Output')
    elif 'pp' in predict and 'h' not in predict:
        outputs = keras.layers.concatenate([ppKL], axis=1, name='Output')
    elif 'pp' not in predict and 'h' in predict:
        outputs = keras.layers.concatenate([hKL], axis=1, name='Output')
        
    model = Model(inputs, outputs)
    
    ## Compile model
    if optimizer == "adam":
        otp = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        otp = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        
    model.compile(optimizer = otp,
                  loss      = loss,
                  metrics   = ['mse', 'mae', 'mape'])
    return model